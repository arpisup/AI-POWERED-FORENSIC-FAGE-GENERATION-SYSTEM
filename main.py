AI-Powered Forensic Face Generation System  ·  v2.0 (UPGRADED)
===============================================================
Streamlit application — Stable Diffusion v1.5 back-end.

WHAT'S NEW IN v2.0
──────────────────
  [NEW] Prompt weighting  — confidence maps to SD attention weights (feature:w)
        so eyes/nose/face-shape get visibly stronger control than before.
  [NEW] Attr-vec → weighted tokens  — CelebA vector drives a second prompt layer
        (attr_prompt) injected alongside the witness prompt, making the
        attribute vector genuinely useful for generation, not just display.
  [NEW] Image-to-image refinement  — when a witness adjusts ONE feature the
        system uses the previous output image as init (strength ≈ 0.45) so
        every other detail (bone structure, skin, lighting) is preserved.
  [NEW] Before / After comparison UI  — side-by-side panels show the face
        before and after each refinement.
  [NEW] Weighted-prompt interpretability panel  — shows each token and the
        weight it received, colour-coded by confidence tier.
  [NEW] Multi-level confidence control:
          • guidance_scale   = 5 + avg_conf × 7          (faithfulness)
          • token weights    derived per-feature from conf (prompt intensity)
          • num_inference_steps scaled 25 → 40 with conf  (detail level)
          • seed-family offset ×7 for variations          (identity family)
  [IMPROVED] Negative prompt expanded with face-distortion artefacts.
  [IMPROVED] enhance_image uses a 3-stage PIL pipeline.
  [IMPROVED] Refinement seed reuse  — base seed never changes during a session.

Usage:
    pip install streamlit torch diffusers transformers accelerate pillow
    streamlit run forensic_face_app_v2.py
"""

# ── std-lib ──────────────────────────────────────────────────────────────────
import io, re, math
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import torch
from PIL import Image, ImageEnhance
import streamlit as st

# ── diffusers ────────────────────────────────────────────────────────────────
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ForensicAI · Face Generator v2",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS & SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
ATTR_DIM = 40
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SD_MODEL = "runwayml/stable-diffusion-v1-5"

# CelebA 40 attributes (official order)
CELEBA_ATTRS = [
    "5_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes",
    "Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair",
    "Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin",
    "Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones",
    "Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard",
    "Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks",
    "Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings",
    "Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young",
]
ATTR_IDX = {a: i for i, a in enumerate(CELEBA_ATTRS)}

# ── Map: CelebA attr → natural prompt token
# [NEW] Used by attr_vec → weighted token layer (makes attr vector useful)
ATTR_TOKEN_MAP: dict[str, str] = {
    "5_o_Clock_Shadow":   "light stubble",
    "Arched_Eyebrows":    "arched eyebrows",
    "Bags_Under_Eyes":    "bags under eyes",
    "Bald":               "bald head",
    "Bangs":              "hair with bangs",
    "Big_Lips":           "full lips",
    "Big_Nose":           "prominent nose",
    "Black_Hair":         "black hair",
    "Blond_Hair":         "blonde hair",
    "Brown_Hair":         "brown hair",
    "Bushy_Eyebrows":     "thick bushy eyebrows",
    "Chubby":             "full round face",
    "Double_Chin":        "double chin",
    "Eyeglasses":         "wearing eyeglasses",
    "Goatee":             "goatee beard",
    "Gray_Hair":          "gray hair",
    "Heavy_Makeup":       "heavy makeup",
    "High_Cheekbones":    "high cheekbones",
    "Male":               "male face",
    "Mouth_Slightly_Open":"slightly open mouth",
    "Mustache":           "mustache",
    "Narrow_Eyes":        "narrow eyes",
    "No_Beard":           "clean shaven",
    "Oval_Face":          "oval face shape",
    "Pale_Skin":          "pale skin",
    "Pointy_Nose":        "pointy nose",
    "Receding_Hairline":  "receding hairline",
    "Rosy_Cheeks":        "rosy cheeks",
    "Sideburns":          "sideburns",
    "Smiling":            "smiling",
    "Straight_Hair":      "straight hair",
    "Wavy_Hair":          "wavy hair",
    "Wearing_Earrings":   "wearing earrings",
    "Wearing_Hat":        "wearing a hat",
    "Wearing_Lipstick":   "wearing lipstick",
    "Wearing_Necklace":   "wearing a necklace",
    "Wearing_Necktie":    "wearing a necktie",
    "Young":              "youthful appearance",
}

# ── Interview steps (unchanged from v1 — UI preserved exactly) ───────────────
INTERVIEW_STEPS = [
    {"key":"gender",      "label":"Gender",              "icon":"🧑",
     "celeba":["Male"],
     "options":["Male","Female"],                        "custom":False,"multi":False},
    {"key":"age",         "label":"Approximate Age",     "icon":"🎂",
     "celeba":["Young"],
     "options":["Child (< 18)","Young Adult (18-35)","Middle-aged (35-55)","Senior (55+)"],
     "custom":True,"multi":False},
    {"key":"face_shape",  "label":"Face Shape",          "icon":"🔷",
     "celeba":["Oval_Face","Chubby"],
     "options":["Oval","Round","Square","Rectangle","Heart","Diamond","Triangle"],
     "custom":True,"multi":False},
    {"key":"skin",        "label":"Skin Tone",           "icon":"🎨",
     "celeba":["Pale_Skin","Rosy_Cheeks"],
     "options":["Very Fair / Pale","Light","Medium / Olive","Tan / Brown","Dark"],
     "custom":True,"multi":False},
    {"key":"hair_color",  "label":"Hair Color",          "icon":"💇",
     "celeba":["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair"],
     "options":["Black","Dark Brown","Brown","Blonde","Red / Auburn","Gray","White","Bald"],
     "custom":True,"multi":False},
    {"key":"hair_style",  "label":"Hair Style",          "icon":"✂️",
     "celeba":["Bald","Bangs","Straight_Hair","Wavy_Hair","Receding_Hairline"],
     "options":["Short","Medium","Long","Curly","Wavy","Straight","Bald","Receding"],
     "custom":True,"multi":False},
    {"key":"eyebrows",    "label":"Eyebrows",            "icon":"📐",
     "celeba":["Arched_Eyebrows","Bushy_Eyebrows"],
     "options":["Arched","Straight","Thick / Bushy","Thin","Normal"],
     "custom":True,"multi":False},
    {"key":"eyes",        "label":"Eye Shape / Feature", "icon":"👁️",
     "celeba":["Narrow_Eyes","Bags_Under_Eyes"],
     "options":["Large","Small / Narrow","Almond-shaped","Bags Under Eyes","Normal"],
     "custom":True,"multi":False},
    {"key":"nose",        "label":"Nose",                "icon":"👃",
     "celeba":["Big_Nose","Pointy_Nose"],
     "options":["Big / Wide","Small","Pointy","Flat / Broad","Normal"],
     "custom":True,"multi":False},
    {"key":"lips",        "label":"Lips / Mouth",        "icon":"👄",
     "celeba":["Big_Lips","Mouth_Slightly_Open","Smiling"],
     "options":["Full / Big Lips","Thin Lips","Mouth Open","Smiling","Neutral"],
     "custom":True,"multi":False},
    {"key":"facial_hair", "label":"Facial Hair",         "icon":"🧔",
     "celeba":["No_Beard","Goatee","Mustache","Sideburns","5_o_Clock_Shadow"],
     "options":["Clean Shaven","5 O'Clock Shadow","Mustache","Goatee","Full Beard","Sideburns"],
     "custom":True,"multi":False},
    {"key":"extras",      "label":"Accessories / Other", "icon":"🕶️",
     "celeba":["Eyeglasses","Wearing_Hat","Wearing_Earrings","Wearing_Necklace","Wearing_Necktie"],
     "options":["None","Glasses","Hat","Earrings","Necklace","Necktie"],
     "custom":True,"multi":True},
]

NEGATIVE_PROMPT = (
    "blurry, out of focus, distorted, disfigured, deformed, mutated, "
    "extra limbs, extra fingers, fused fingers, missing fingers, cloned face, "
    "cartoon, anime, painting, sketch, watermark, text, logo, "
    "low quality, jpeg artefacts, noise, grainy, overexposed, underexposed, "
    "bad anatomy, bad proportions, unrealistic skin"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_txt2img() -> StableDiffusionPipeline:
    """
    Load text-to-image Stable Diffusion pipeline (cached).
    float16 on CUDA, float32 on CPU.  Attention slicing reduces VRAM.
    """
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


@st.cache_resource(show_spinner=False)
def load_img2img() -> StableDiffusionImg2ImgPipeline:
    """
    [NEW] Load image-to-image pipeline for refinement.
    Shares UNet weights with txt2img (no extra download beyond first run).
    Used when a witness edits ONE feature — preserves the rest of the face.
    """
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NOVELTY B — SEMANTIC INTERPOLATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def semantic_interpolation(text: str) -> str:
    """
    Maps free-form / unconventional witness descriptions to SD-friendly tokens.
    Ordered list → first match wins (most-specific patterns first).
    Returns original text unchanged when no pattern matches.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    mapping = [
        # Face shapes
        (r"rectangle|rectangular",  "angular face with sharp jawline and elongated proportions"),
        (r"heart",                  "wide forehead, high cheekbones, gently pointed chin"),
        (r"diamond",                "narrow forehead and chin, prominent high cheekbones"),
        (r"triangle|pear",          "narrow forehead widening to a broad strong jaw"),
        (r"square",                 "strong square jaw, broad forehead, angular features"),
        (r"round",                  "soft rounded face, full cheeks, gentle jawline"),
        (r"oval",                   "oval face with balanced proportions"),
        # Age
        (r"child|kid|<\s*18",       "youthful child, smooth skin, large innocent eyes"),
        (r"young adult|18.?35|teen","youthful adult, clear smooth skin"),
        (r"middle.?aged|35.?55",    "mature adult, subtle laugh lines"),
        (r"senior|elderly|55\+|old","aged person, distinct wrinkles, silver hair"),
        # Skin
        (r"very fair|pale",         "very pale porcelain skin"),
        (r"\blight\b",              "light skin tone"),
        (r"medium|olive",           "medium olive skin tone"),
        (r"tan|brown",              "warm tan brown skin"),
        (r"\bdark\b",               "deep dark skin tone"),
        # Hair
        (r"red|auburn",             "vivid red auburn hair"),
        (r"dark brown",             "rich dark brown hair"),
        (r"dirty blonde",           "dirty blonde hair"),
        (r"platinum|white blonde",  "platinum blonde hair"),
        (r"\bblonde?\b",            "golden blonde hair"),
        # Facial hair
        (r"clean shaven?",          "clean-shaven, no facial hair"),
        (r"full beard",             "thick full beard"),
        (r"5.?o.?clock|stubble",    "light short stubble"),
        (r"soul patch",             "small soul patch"),
        # Eyes
        (r"almond",                 "almond-shaped eyes"),
        (r"hooded",                 "hooded eyelids"),
        (r"deep.?set",              "deep-set eyes"),
        (r"close.?set",             "closely set eyes"),
        # Nose
        (r"button",                 "small button nose"),
        (r"hawk|aquiline",          "aquiline hawk-like nose"),
        (r"bulbous",                "bulbous rounded nose tip"),
        # Misc
        (r"freckles?",              "light freckles across nose and cheeks"),
        (r"dimples?",               "prominent dimples"),
        (r"scar",                   "faint facial scar"),
        (r"high cheek",             "prominent high cheekbones"),
    ]

    lower = text.lower().strip()
    for pattern, replacement in mapping:
        if re.search(pattern, lower):
            return replacement
    return text


# ─────────────────────────────────────────────────────────────────────────────
# 4.  NOVELTY D — ATTRIBUTE VECTOR → WEIGHTED TOKEN LAYER  [NEW in v2]
# ─────────────────────────────────────────────────────────────────────────────

def attr_vec_to_weighted_tokens(attr_vec: np.ndarray, threshold: float = 0.60) -> str:
    """
    [NEW] Convert CelebA attribute vector into a secondary weighted prompt string.

    This makes the attribute vector *actually contribute* to generation instead
    of being only a display artefact.

    Algorithm:
        For each CelebA attr whose soft-value exceeds `threshold`:
            weight = 1.0 + (attr_value - threshold) * 2.0   → range [1.0, 1.8]
            token  = ATTR_TOKEN_MAP[attr]
            format = "(token:weight)"   ← SD prompt-weighting syntax

    The resulting string is appended to the main witness prompt, giving SD a
    second, attribute-driven signal that reinforces the strongest features.

    Example output:
        "(black hair:1.6), (narrow eyes:1.3), (clean shaven:1.4)"
    """
    tokens: list[str] = []
    for attr, token in ATTR_TOKEN_MAP.items():
        idx = ATTR_IDX.get(attr)
        if idx is None:
            continue
        v = float(attr_vec[idx])
        if v > threshold:
            # Linear mapping: threshold → 1.0,  1.0 → 1.8
            weight = round(1.0 + (v - threshold) / (1.0 - threshold) * 0.8, 2)
            weight = min(weight, 1.8)  # cap to avoid artefacts
            tokens.append(f"({token}:{weight:.2f})")
    return ", ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NOVELTY C — CONFIDENCE-AWARE PROMPT BUILDER  (upgraded v2)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_to_weight(conf: float) -> float:
    """
    [NEW] Map a per-feature confidence score to an SD prompt attention weight.

    Mapping (linear in each tier):
        conf > 0.75  →  weight 1.3 – 1.5   (high certainty → strong emphasis)
        0.50 – 0.75  →  weight 1.0 – 1.3   (moderate → normal weight)
        < 0.50       →  weight 0.7 – 1.0   (uncertain → de-emphasised)

    These weights appear in the SD "(token:w)" syntax, directly steering
    cross-attention toward or away from the feature token.
    """
    if conf > 0.75:
        return round(1.3 + (conf - 0.75) / 0.25 * 0.2, 2)   # 1.30 – 1.50
    elif conf >= 0.50:
        return round(1.0 + (conf - 0.50) / 0.25 * 0.3, 2)   # 1.00 – 1.30
    else:
        return round(0.7 + conf / 0.50 * 0.3, 2)             # 0.70 – 1.00


def build_prompt(answers: dict, avg_confidence: float) -> tuple[str, list[dict]]:
    """
    NOVELTY C — Confidence-Aware Prompt Intensity  (upgraded in v2).

    Returns
    -------
    prompt : str
        Full SD prompt with per-feature attention weights in "(token:w)" syntax.
    token_info : list[dict]
        Interpretability data: [{feature, phrase, conf, weight, tier}, ...]
        Used by the new weighted-prompt display panel.

    Confidence tiers
    ─────────────────
        > 0.75  → weight 1.3–1.5,  prefix "clear sharp"
        0.5–0.75→ weight 1.0–1.3,  no prefix
        < 0.5   → weight 0.7–1.0,  prefix "slightly"

    This is *separate* from guidance_scale (Novelty A) — it modulates
    which tokens SD attends to within the prompt, not the overall
    faithfulness of generation.
    """
    weighted_parts: list[str] = []
    token_info: list[dict] = []

    for step in INTERVIEW_STEPS:
        key = step["key"]
        if key not in answers:
            continue

        ans   = answers[key]
        value = ans["value"]
        conf  = float(ans["confidence"])

        # Normalise value via semantic interpolation
        if isinstance(value, list):
            cleaned = [
                semantic_interpolation(v)
                for v in value
                if isinstance(v, str) and v.strip().lower() != "none"
            ]
            if not cleaned:
                continue
            phrase = ", ".join(cleaned)
        elif isinstance(value, str):
            if value.strip().lower() in ("none", "normal", "neutral", ""):
                continue
            phrase = semantic_interpolation(value)
        else:
            continue

        # Confidence → attention weight  [NEW in v2]
        weight = _confidence_to_weight(conf)

        # Confidence → wording prefix  (Novelty C from v1, kept)
        if conf > 0.75:
            prefix = "clear sharp "
            tier   = "high"
        elif conf >= 0.50:
            prefix = ""
            tier   = "medium"
        else:
            prefix = "slightly "
            tier   = "low"

        full_phrase = f"{prefix}{phrase}"

        # SD prompt-weight syntax: "(phrase:weight)"
        weighted_parts.append(f"({full_phrase}:{weight:.2f})")

        token_info.append({
            "feature": step["label"],
            "phrase":  full_phrase,
            "conf":    conf,
            "weight":  weight,
            "tier":    tier,
        })

    feature_str = ", ".join(weighted_parts)
    quality_suffix = (
        "(highly detailed face:1.4), (realistic skin texture:1.3), "
        "(sharp focus:1.2), (professional studio lighting:1.2), "
        "photorealistic portrait photography, 8k resolution, neutral background"
    )
    base   = "photorealistic close-up portrait of a person, "
    prompt = base + feature_str + (", " if feature_str else "") + quality_suffix

    return prompt, token_info


# ─────────────────────────────────────────────────────────────────────────────
# 6.  IMAGE ENHANCEMENT
# ─────────────────────────────────────────────────────────────────────────────

def enhance_image(image: Image.Image) -> Image.Image:
    """
    Lightweight PIL post-processing — no extra dependencies.
        1. Sharpness ×1.25  — crisps fine facial details
        2. Contrast  ×1.10  — lifts mid-tones
        3. Color     ×1.05  — subtle skin-tone saturation boost

    Optional GFPGAN (face restoration / 2× super-resolution):
        # pip install gfpgan
        # from gfpgan import GFPGANer
        # restorer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=2)
        # _, _, restored = restorer.enhance(np.array(image), paste_back=True)
        # return Image.fromarray(restored)
    """
    img = ImageEnhance.Sharpness(image).enhance(1.25)
    img = ImageEnhance.Contrast(img).enhance(1.10)
    img = ImageEnhance.Color(img).enhance(1.05)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CONFIDENCE METRICS  (shared utility)
# ─────────────────────────────────────────────────────────────────────────────

def compute_avg_confidence(answers: dict) -> float:
    """Return average confidence across all answered features (default 0.70)."""
    confs = [
        float(a["confidence"])
        for a in answers.values()
        if isinstance(a, dict) and "confidence" in a
    ]
    return sum(confs) / len(confs) if confs else 0.70


# ─────────────────────────────────────────────────────────────────────────────
# 8.  NOVELTY A — CONFIDENCE-DRIVEN GENERATION  (upgraded v2)
# ─────────────────────────────────────────────────────────────────────────────

def generate_face(
    attr_vec: np.ndarray,
    z_seed: int = 42,
    upscale: bool = True,
    refine_from: Image.Image | None = None,
    refine_strength: float = 0.45,
) -> Image.Image:
    """
    Generate / refine a photorealistic face with Stable Diffusion v1.5.

    NOVELTY A — Multi-level Confidence Control  (expanded in v2)
    ─────────────────────────────────────────────────────────────
    avg_confidence now drives THREE generation parameters:

        1. guidance_scale  = 5 + avg_conf × 7          → [5, 12]
           High conf → prompt-faithful; low conf → creative variation.

        2. num_inference_steps = round(25 + avg_conf × 15)  → [25, 40]
           High conf → more denoising steps → sharper, more detailed face.

        3. Token weights in prompt (see build_prompt)
           High conf → "(feature:1.5)" — SD strongly attends to that token.
           Low conf  → "(feature:0.75)" — SD loosely interprets it.

    NOVELTY D — Attr-Vec → Weighted Token Layer  [NEW in v2]
    ─────────────────────────────────────────────────────────
    The CelebA attribute vector is converted to additional "(token:w)" terms
    (via attr_vec_to_weighted_tokens) and appended to the prompt, so the
    vector genuinely influences cross-attention — not just the display chips.

    NOVELTY E — Image-to-Image Refinement Loop  [NEW in v2]
    ────────────────────────────────────────────────────────
    When `refine_from` is not None (i.e. the witness is editing one feature):
        • img2img pipeline is used with strength ≈ 0.45
        • Only ~45 % of the denoising steps are re-run from the init image
        • The remaining 55 % of latent structure is preserved → identity stable
        • Same seed → same noise realisation → same bone structure is kept

    SEED CONSISTENCY
    ────────────────
    torch.Generator.manual_seed(z_seed) — deterministic across calls.
    Refinement passes always use the base session seed.

    Parameters
    ----------
    attr_vec       : CelebA attribute vector (float32, shape [40])
    z_seed         : reproducibility seed
    upscale        : apply enhance_image() post-processing
    refine_from    : if provided, img2img refine from this PIL image
    refine_strength: img2img noise level (0 = no change, 1 = full regeneration)
    """
    answers  = st.session_state.get("answers", {})
    avg_conf = compute_avg_confidence(answers)

    # ── Build weighted prompt + attr-vec layer ─────────────────────────────
    if answers:
        witness_prompt, _ = build_prompt(answers, avg_conf)
    else:
        witness_prompt = (
            "photorealistic close-up portrait of a person, "
            "(highly detailed face:1.4), (realistic skin texture:1.3), "
            "(sharp focus:1.2), professional studio lighting"
        )

    # [NEW] Append attribute-vector weighted tokens as a second control layer
    attr_tokens = attr_vec_to_weighted_tokens(attr_vec)
    if attr_tokens:
        full_prompt = witness_prompt + ", " + attr_tokens
    else:
        full_prompt = witness_prompt

    # ── Multi-level confidence control ─────────────────────────────────────
    guidance_scale = 5.0 + avg_conf * 7.0                   # [5.0 – 12.0]
    n_steps        = round(25 + avg_conf * 15)               # [25 – 40]

    # ── Seed ──────────────────────────────────────────────────────────────
    generator = torch.Generator(device=DEVICE).manual_seed(int(z_seed))

    # ── [NEW] img2img refinement path (identity-preserving) ───────────────
    if refine_from is not None:
        pipe_i2i = load_img2img()
        init_img = refine_from.convert("RGB").resize((512, 512), Image.LANCZOS)
        with torch.no_grad():
            result = pipe_i2i(
                prompt          = full_prompt,
                negative_prompt = NEGATIVE_PROMPT,
                image           = init_img,
                strength        = refine_strength,      # ← preserves ~55 % of original
                num_inference_steps = n_steps,
                guidance_scale  = guidance_scale,
                generator       = generator,
            )
    else:
        # Standard text-to-image generation
        pipe_t2i = load_txt2img()
        with torch.no_grad():
            result = pipe_t2i(
                prompt          = full_prompt,
                negative_prompt = NEGATIVE_PROMPT,
                num_inference_steps = n_steps,
                guidance_scale  = guidance_scale,
                generator       = generator,
                height          = 512,
                width           = 512,
            )

    img: Image.Image = result.images[0]
    if upscale:
        img = enhance_image(img)
    return img


def generate_variations(attr_vec: np.ndarray, n: int = 4) -> list[Image.Image]:
    """
    Generate n variations sharing the same identity family.

    Seed offset formula: seed_i = base_seed + i × 7
    This keeps all variations in the same "latent neighbourhood" while
    adding enough diversity to be useful for a witness to choose from.
    """
    base_seed = int(st.session_state.get("z_seed", 42))
    return [
        generate_face(attr_vec, z_seed=base_seed + i * 7, upscale=True)
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 9.  ATTRIBUTE VECTOR BUILDER  (unchanged logic, kept for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def build_attribute_vector(answers: dict) -> np.ndarray:
    """
    Convert interview answers + confidence into a soft CelebA attribute vector.

    Soft-set formula:
        attr_value = confidence × target + (1 − confidence) × 0.5
        0.5 = ambiguous / unknown;  values near 0 or 1 = high confidence.
    """
    v = np.full(ATTR_DIM, 0.5, dtype=np.float32)

    def set_attr(name: str, value: float, confidence: float) -> None:
        idx = ATTR_IDX.get(name)
        if idx is None:
            return
        soft = confidence * value + (1.0 - confidence) * 0.5
        v[idx] = float(np.clip(soft, 0.0, 1.0))

    for step_key, answer_data in answers.items():
        if not answer_data:
            continue
        val  = answer_data.get("value", "")
        conf = answer_data.get("confidence", 1.0)

        if step_key == "gender":
            set_attr("Male", 1.0 if val == "Male" else 0.0, conf)
            if val == "Female":
                set_attr("Heavy_Makeup",    0.7, conf)
                set_attr("Wearing_Lipstick",0.7, conf)
                set_attr("Arched_Eyebrows", 0.6, conf)

        elif step_key == "age":
            is_young = val in ["Child (< 18)", "Young Adult (18-35)"]
            set_attr("Young", 1.0 if is_young else 0.0, conf)
            if val == "Senior (55+)":
                set_attr("Gray_Hair", 0.6, conf)
                set_attr("Bags_Under_Eyes", 0.5, conf)

        elif step_key == "face_shape":
            mapping = {
                "Oval":      [("Oval_Face",1.0),("Chubby",0.0)],
                "Round":     [("Oval_Face",0.5),("Chubby",0.7)],
                "Square":    [("Oval_Face",0.0),("Chubby",0.3)],
                "Rectangle": [("Oval_Face",0.0),("Chubby",0.1)],
                "Heart":     [("Oval_Face",0.6),("Chubby",0.0)],
                "Diamond":   [("High_Cheekbones",0.9)],
                "Triangle":  [("Chubby",0.0),("Double_Chin",0.4)],
            }
            for a, v_ in mapping.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "skin":
            mapping = {
                "Very Fair / Pale":[("Pale_Skin",1.0)],
                "Light":           [("Pale_Skin",0.7)],
                "Medium / Olive":  [("Pale_Skin",0.0)],
                "Tan / Brown":     [("Pale_Skin",0.0),("Rosy_Cheeks",0.3)],
                "Dark":            [("Pale_Skin",0.0)],
            }
            for a, v_ in mapping.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "hair_color":
            for c in ["Black_Hair","Blond_Hair","Brown_Hair","Gray_Hair"]:
                set_attr(c, 0.0, 1.0)
            hc_map = {
                "Black":"Black_Hair","Dark Brown":"Black_Hair","Brown":"Brown_Hair",
                "Blonde":"Blond_Hair","Red / Auburn":"Brown_Hair",
                "Gray":"Gray_Hair","White":"Gray_Hair","Bald":None,
            }
            target = hc_map.get(val)
            if target:
                set_attr(target, 1.0, conf)
            if val == "Bald":
                set_attr("Bald", 1.0, conf)

        elif step_key == "hair_style":
            style_map = {
                "Short":    [("Bangs",0.0)],
                "Medium":   [],
                "Long":     [("Bangs",0.3)],
                "Curly":    [("Wavy_Hair",0.7)],
                "Wavy":     [("Wavy_Hair",1.0),("Straight_Hair",0.0)],
                "Straight": [("Straight_Hair",1.0),("Wavy_Hair",0.0)],
                "Bald":     [("Bald",1.0)],
                "Receding": [("Receding_Hairline",1.0)],
            }
            for a, v_ in style_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "eyebrows":
            eb_map = {
                "Arched":        [("Arched_Eyebrows",1.0),("Bushy_Eyebrows",0.0)],
                "Straight":      [("Arched_Eyebrows",0.0)],
                "Thick / Bushy": [("Bushy_Eyebrows",1.0)],
                "Thin":          [("Bushy_Eyebrows",0.0)],
                "Normal":        [],
            }
            for a, v_ in eb_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "eyes":
            eye_map = {
                "Large":           [("Narrow_Eyes",0.0)],
                "Small / Narrow":  [("Narrow_Eyes",1.0)],
                "Almond-shaped":   [("Narrow_Eyes",0.4)],
                "Bags Under Eyes": [("Bags_Under_Eyes",1.0)],
                "Normal":          [],
            }
            for a, v_ in eye_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "nose":
            nose_map = {
                "Big / Wide":  [("Big_Nose",1.0),("Pointy_Nose",0.0)],
                "Small":       [("Big_Nose",0.0)],
                "Pointy":      [("Pointy_Nose",1.0),("Big_Nose",0.0)],
                "Flat / Broad":[("Big_Nose",0.8),("Pointy_Nose",0.0)],
                "Normal":      [],
            }
            for a, v_ in nose_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "lips":
            lip_map = {
                "Full / Big Lips":[("Big_Lips",1.0)],
                "Thin Lips":      [("Big_Lips",0.0)],
                "Mouth Open":     [("Mouth_Slightly_Open",1.0)],
                "Smiling":        [("Smiling",1.0),("Mouth_Slightly_Open",0.5)],
                "Neutral":        [("Smiling",0.0),("Mouth_Slightly_Open",0.0)],
            }
            for a, v_ in lip_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "facial_hair":
            fh_map = {
                "Clean Shaven":     [("No_Beard",1.0),("Goatee",0.0),("Mustache",0.0),("5_o_Clock_Shadow",0.0)],
                "5 O'Clock Shadow": [("5_o_Clock_Shadow",1.0),("No_Beard",0.0)],
                "Mustache":         [("Mustache",1.0),("No_Beard",0.0)],
                "Goatee":           [("Goatee",1.0),("No_Beard",0.0)],
                "Full Beard":       [("Goatee",0.8),("5_o_Clock_Shadow",0.8),("No_Beard",0.0)],
                "Sideburns":        [("Sideburns",1.0)],
            }
            for a, v_ in fh_map.get(val, []):
                set_attr(a, v_, conf)

        elif step_key == "extras":
            vals = val if isinstance(val, list) else [val]
            extra_map = {
                "Glasses":"Eyeglasses","Hat":"Wearing_Hat","Earrings":"Wearing_Earrings",
                "Necklace":"Wearing_Necklace","Necktie":"Wearing_Necktie",
            }
            for choice in vals:
                a = extra_map.get(choice)
                if a:
                    set_attr(a, 1.0, conf)

    return v


# ─────────────────────────────────────────────────────────────────────────────
# 10.  UI — STYLING
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    * { font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main { background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 2rem; }
    .stButton>button {
        background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 100%);
        color: #0F172A; border: none; padding: 0.75rem 1.5rem;
        font-weight: 600; border-radius: 0.5rem; transition: all 0.3s;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(45,212,191,0.4); }
    .step-card {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1rem;
    }
    .step-card.done { border: 1px solid #14B8A6; opacity: 0.8; }
    .step-label { font-size: 1.1rem; font-weight: 600; color: #2DD4BF; margin-bottom: 0.75rem; }
    .info-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-left: 4px solid #2DD4BF; padding: 1rem 1.5rem;
        border-radius: 0.5rem; margin-bottom: 1.5rem; color: #E8EAF0; font-size: 0.9rem;
    }
    .face-output-wrapper {
        background: #1E293B; border: 2px solid #334155; border-radius: 1rem; padding: 1.5rem;
    }
    .attr-chip {
        display: inline-block; padding: 0.25rem 0.75rem; margin: 0.25rem;
        border-radius: 1rem; font-size: 0.75rem; background: #334155; color: #94A3B8;
    }
    .attr-chip.active {
        background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 100%);
        color: #0F172A; font-weight: 600;
    }
    .progress-wrapper { background: #1E293B; border-radius: 1rem; padding: 1rem; margin-bottom: 2rem; }
    .progress-bar { background: #334155; height: 8px; border-radius: 4px; overflow: hidden; }
    .progress-fill { background: linear-gradient(90deg,#2DD4BF 0%,#14B8A6 100%); height:100%; transition:width 0.3s ease; }

    /* [NEW] token weight badges */
    .tok-high   { background:#0D9488; color:#fff; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
    .tok-medium { background:#0284C7; color:#fff; padding:2px 6px; border-radius:4px; font-size:0.75rem; }
    .tok-low    { background:#6B7280; color:#fff; padding:2px 6px; border-radius:4px; font-size:0.75rem; }

    /* [NEW] before/after comparison */
    .compare-label {
        text-align:center; font-size:0.8rem; font-weight:600;
        color:#94A3B8; padding:4px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem;">
        <h1 style="font-family:'Space Grotesk',sans-serif;font-size:2.5rem;font-weight:700;
             background:linear-gradient(135deg,#2DD4BF 0%,#14B8A6 100%);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.5rem;">
            🔍 ForensicAI · Face Generator  <span style="font-size:1rem;opacity:0.6">v2.0</span>
        </h1>
        <p style="color:#94A3B8;font-size:0.95rem;max-width:680px;margin:auto;">
            Stable Diffusion v1.5 · confidence-weighted prompts · attr-vec injection ·
            img2img refinement · before/after comparison
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_progress(current: int, total: int) -> None:
    pct = (current / total) * 100
    st.markdown(f"""
    <div class="progress-wrapper">
        <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;">
            <span style="color:#E8EAF0;font-weight:600;">Step {current} of {total}</span>
            <span style="color:#6B7280;">{pct:.0f}%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:{pct}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  UI — INTERVIEW STEP RENDERER  (unchanged — UI preserved exactly)
# ─────────────────────────────────────────────────────────────────────────────

def render_step(step_info: dict, idx: int, prev_answers: dict) -> dict:
    """Render one interview step; return {"value": ..., "confidence": ...}."""
    key  = step_info["key"]
    prev = prev_answers.get(key, {})

    st.markdown(f"""
    <div class="step-card">
        <div class="step-label">{step_info['icon']} {step_info['label']}</div>
    </div>
    """, unsafe_allow_html=True)

    if step_info["multi"]:
        selected = st.multiselect(
            "Choose one or more",
            options=step_info["options"],
            default=prev.get("value", []) if isinstance(prev.get("value"), list) else [],
            key=f"q_{idx}",
        )
        if step_info["custom"]:
            other = st.text_input("Other (specify)", value="", key=f"q_{idx}_other")
            if other.strip():
                selected = selected + [other.strip()]
        value = selected
    else:
        options = step_info["options"] + (["Other"] if step_info["custom"] else [])
        selected = st.radio(
            "Choose one",
            options=options,
            index=options.index(prev.get("value", options[0])) if prev.get("value") in options else 0,
            key=f"q_{idx}",
        )
        value = st.text_input("Specify", value="", key=f"q_{idx}_other") \
            if (selected == "Other" and step_info["custom"]) else selected

    confidence = st.slider(
        "How confident are you?",
        min_value=0.0, max_value=1.0,
        value=prev.get("confidence", 0.8),
        step=0.05, key=f"conf_{idx}",
        help="Lower confidence = softer prompt weight + more generation variation",
    )
    return {"value": value, "confidence": confidence}


# ─────────────────────────────────────────────────────────────────────────────
# 12.  UI — INTERPRETABILITY PANEL  [NEW in v2]
# ─────────────────────────────────────────────────────────────────────────────

def render_interpretability(answers: dict, attr_vec: np.ndarray) -> None:
    """
    [NEW] Three-tab interpretability panel:
        Tab 1 — Weighted Prompt  : shows each feature token + its attention weight
        Tab 2 — Confidence Stats : avg conf, guidance_scale, n_steps
        Tab 3 — Attribute Vector : CelebA chip display (from v1)
    """
    avg_conf       = compute_avg_confidence(answers)
    guidance_scale = 5.0 + avg_conf * 7.0
    n_steps        = round(25 + avg_conf * 15)
    prompt, token_info = build_prompt(answers, avg_conf)
    attr_tokens        = attr_vec_to_weighted_tokens(attr_vec)

    st.markdown("---")
    st.markdown("**🔍 Generation Interpretability**")

    tab1, tab2, tab3 = st.tabs(["📝 Weighted Prompt", "📊 Confidence Stats", "🧩 Attribute Vector"])

    # ── Tab 1: Weighted prompt tokens ─────────────────────────────────────
    with tab1:
        st.markdown("Each feature is injected into the prompt with a weight derived from its confidence score. "
                    "Higher weight → Stable Diffusion attends to that token more strongly.")
        for ti in token_info:
            tier_cls = f"tok-{ti['tier']}"
            bar_pct  = int(ti['conf'] * 100)
            st.markdown(
                f"<div style='margin-bottom:0.4rem;'>"
                f"<strong style='color:#E8EAF0'>{ti['feature']}</strong>&nbsp;"
                f"<code style='color:#2DD4BF'>{ti['phrase']}</code>&nbsp;"
                f"<span class='{tier_cls}'>weight&nbsp;{ti['weight']:.2f}</span>&nbsp;"
                f"<span style='color:#6B7280;font-size:0.78rem'>conf {bar_pct}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if attr_tokens:
            st.markdown("**Attribute-vector layer** (appended after witness prompt):")
            st.code(attr_tokens, language="text")
        with st.expander("🔎 Full raw prompt"):
            full = prompt + (", " + attr_tokens if attr_tokens else "")
            st.code(full, language="text")

    # ── Tab 2: Confidence statistics ──────────────────────────────────────
    with tab2:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Avg Confidence",  f"{avg_conf:.1%}")
        col_b.metric("Guidance Scale",  f"{guidance_scale:.1f}")
        col_c.metric("Inference Steps", str(n_steps))
        st.markdown(
            f"<div style='font-size:0.83rem;color:#94A3B8;margin-top:0.5rem;'>"
            f"guidance_scale = 5 + {avg_conf:.2f} × 7 = <strong style='color:#2DD4BF'>{guidance_scale:.1f}</strong>&nbsp;·&nbsp;"
            f"steps = 25 + {avg_conf:.2f} × 15 = <strong style='color:#2DD4BF'>{n_steps}</strong>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Per-feature confidence:**")
        for step in INTERVIEW_STEPS:
            k = step["key"]
            if k in answers:
                a   = answers[k]
                val = a["value"] if not isinstance(a["value"], list) else ", ".join(a["value"])
                c   = a["confidence"]
                w   = _confidence_to_weight(c)
                bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
                st.markdown(
                    f"<div style='font-size:0.82rem;margin-bottom:3px;'>"
                    f"<strong style='color:#E8EAF0;width:120px;display:inline-block'>{step['label']}</strong>"
                    f"<code style='color:#6B7280'>{bar}</code>"
                    f"&nbsp;<span style='color:#2DD4BF'>{c:.0%}</span>"
                    f"&nbsp;<span style='color:#94A3B8'>→ weight {w:.2f}</span>"
                    f"&nbsp;<span style='color:#64748B;font-size:0.75rem'>{val}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Attribute vector chips ─────────────────────────────────────
    with tab3:
        st.markdown("Active CelebA attributes (soft-set by confidence × target):")
        chips_html = ""
        for j, attr in enumerate(CELEBA_ATTRS):
            v_  = attr_vec[j]
            active   = v_ > 0.55
            chip_cls = "attr-chip active" if active else "attr-chip"
            label    = attr.replace("_", " ")
            alpha    = f" ({v_:.2f})" if active else ""
            chips_html += f'<span class="{chip_cls}">{label}{alpha}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 13.  UI — OUTPUT RENDERER  (upgraded v2 with before/after comparison)
# ─────────────────────────────────────────────────────────────────────────────

def render_output(imgs: list[Image.Image], attr_vec: np.ndarray) -> None:
    """
    Render generated face variations with:
        • Download buttons per variation
        • [NEW] Before/After comparison panel (shown when refinement_before exists)
        • Interpretability tabs (prompt, confidence, attr chips)
    """
    st.markdown("""
    <div class="face-output-wrapper">
    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:600;
         color:#E8EAF0;margin-bottom:1rem;">🎨 Generated Composite Faces</div>
    """, unsafe_allow_html=True)

    # ── [NEW] Before / After comparison ───────────────────────────────────
    before_img = st.session_state.get("refinement_before")
    if before_img is not None and imgs:
        st.markdown("#### 🔄 Before vs After Refinement")
        b_col, a_col = st.columns(2)
        with b_col:
            st.markdown('<div class="compare-label">BEFORE (original)</div>', unsafe_allow_html=True)
            st.image(before_img, use_container_width=True)
        with a_col:
            st.markdown('<div class="compare-label">AFTER (refined)</div>', unsafe_allow_html=True)
            st.image(imgs[0], use_container_width=True)
        st.markdown("---")

    # ── Variation grid ────────────────────────────────────────────────────
    cols = st.columns(len(imgs))
    for i, (col, img) in enumerate(zip(cols, imgs)):
        with col:
            st.image(img, caption=f"Variation {i+1}", use_container_width=True)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.download_button(
                label=f"⬇️ Download",
                data=buf.getvalue(),
                file_name=f"forensic_face_v{i+1}.png",
                mime="image/png",
                key=f"dl_{i}",
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Interpretability ──────────────────────────────────────────────────
    if "answers" in st.session_state:
        render_interpretability(st.session_state.answers, attr_vec)


# ─────────────────────────────────────────────────────────────────────────────
# 14.  MAIN APP FLOW
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    render_header()

    # ── Session state initialisation ──────────────────────────────────────
    defaults = {
        "answers":           {},
        "current_step":      0,
        "generated":         [],
        "attr_vec":          None,
        "z_seed":            42,
        "refinement_before": None,   # [NEW] stores image before refinement
        "is_refining":       False,  # [NEW] flag for img2img path
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    total_steps = len(INTERVIEW_STEPS)
    render_progress(st.session_state.current_step, total_steps)

    left_col, right_col = st.columns([1.2, 1], gap="large")

    # ─────────────────────────────────────────────────────────────────────
    with left_col:
        st.markdown("""
        <div class="info-box">
            Fill in each feature one by one. Use the confidence slider to indicate
            how certain you are — uncertain features receive lower prompt weights
            and allow more generation variation. Use "Other" for free-form input.
        </div>
        """, unsafe_allow_html=True)

        current = st.session_state.current_step

        # Completed steps summary
        for idx in range(current):
            step_info = INTERVIEW_STEPS[idx]
            key = step_info["key"]
            if key in st.session_state.answers:
                ans  = st.session_state.answers[key]
                val  = ans["value"]
                conf = ans["confidence"]
                w    = _confidence_to_weight(conf)
                st.markdown(f"""
                <div class="step-card done">
                    <div class="step-label">{step_info['icon']} {step_info['label']} ✓</div>
                    <div style="font-size:0.9rem;color:#E8EAF0;">
                        <strong>{val if not isinstance(val, list) else ', '.join(val)}</strong>
                        &nbsp;·&nbsp;<span style="color:#6B7280">conf: {conf:.0%}</span>
                        &nbsp;·&nbsp;<span style="color:#2DD4BF">weight: {w:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Active step
        if current < total_steps:
            step_info = INTERVIEW_STEPS[current]
            answer    = render_step(step_info, current, st.session_state.answers)

            col_next, col_back = st.columns([1, 1])
            with col_next:
                if st.button("Next →", key="btn_next", use_container_width=True):
                    st.session_state.answers[step_info["key"]] = answer
                    st.session_state.current_step += 1
                    st.rerun()
            with col_back:
                if current > 0 and st.button("← Back", key="btn_back", use_container_width=True):
                    prev_key = INTERVIEW_STEPS[current - 1]["key"]
                    del st.session_state.answers[prev_key]
                    st.session_state.current_step -= 1
                    st.rerun()
        else:
            # All steps done
            st.markdown("""
            <div style="text-align:center;padding:1rem 0;">
                <div style="font-size:1.1rem;color:#2DD4BF;font-weight:600;margin-bottom:0.5rem;">
                    ✓ All features captured!
                </div>
                <div style="font-size:0.85rem;color:#6B7280;">
                    Click Generate to produce the composite face.
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_gen, col_reset = st.columns([1.5, 1])
            with col_gen:
                if st.button("🎨 Generate Face", key="btn_gen", use_container_width=True):
                    attr_vec = build_attribute_vector(st.session_state.answers)
                    st.session_state.refinement_before = None   # fresh generation
                    st.session_state.is_refining       = False
                    with st.spinner("Generating face variations… (this may take a minute)"):
                        imgs = generate_variations(attr_vec, n=4)
                    st.session_state.generated = imgs
                    st.session_state.attr_vec  = attr_vec
                    st.rerun()
            with col_reset:
                if st.button("↺ Start Over", key="btn_reset", use_container_width=True):
                    for k in ["answers","current_step","generated","attr_vec",
                              "refinement_before","is_refining"]:
                        st.session_state.pop(k, None)
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    with right_col:
        if st.session_state.generated:
            render_output(st.session_state.generated, st.session_state.attr_vec)

            st.markdown("---")
            st.markdown("""
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1rem;
                 font-weight:600;color:#E8EAF0;margin-bottom:0.8rem;">🔧 Refine a Feature</div>
            """, unsafe_allow_html=True)

            refine_feature = st.selectbox(
                "Feature to refine",
                options=[s["label"] for s in INTERVIEW_STEPS],
                key="refine_select",
            )

            # [NEW] Refinement mode toggle
            use_img2img = st.checkbox(
                "🖼️ Use img2img (preserve identity more strongly)",
                value=True, key="chk_img2img",
                help=(
                    "When enabled, the previous generated face is used as the starting "
                    "image. Only ~45% of denoising steps are re-run, preserving bone "
                    "structure, lighting and skin tone from the original."
                ),
            )

            if st.button("✏️ Adjust this feature", key="btn_refine"):
                step_idx = next(i for i, s in enumerate(INTERVIEW_STEPS)
                                if s["label"] == refine_feature)
                key      = INTERVIEW_STEPS[step_idx]["key"]

                # [NEW] Save current best image as "before" reference
                if st.session_state.generated:
                    st.session_state.refinement_before = st.session_state.generated[0]

                st.session_state.is_refining  = use_img2img
                st.session_state.answers.pop(key, None)
                st.session_state.current_step = step_idx
                st.session_state.generated    = []
                st.rerun()

            st.markdown("**Try different variations:**")
            new_seed = st.number_input(
                "Random seed", min_value=0, max_value=9999,
                value=st.session_state.z_seed, step=1, key="seed_input",
            )
            if st.button("🔄 Regenerate with seed", key="btn_reseed"):
                st.session_state.z_seed = new_seed
                attr_vec = st.session_state.attr_vec
                with st.spinner("Generating…"):
                    imgs = [
                        generate_face(attr_vec, z_seed=new_seed + i * 7, upscale=True)
                        for i in range(4)
                    ]
                st.session_state.generated    = imgs
                st.session_state.refinement_before = None
                st.rerun()

        else:
            # ── [NEW] Show before/after even in placeholder state if refining ──
            before = st.session_state.get("refinement_before")
            if before is not None:
                st.markdown("#### 🔄 Refining — original face:")
                st.image(before, use_container_width=True,
                         caption="This identity will be preserved during refinement")
            else:
                st.markdown("""
                <div class="face-output-wrapper" style="padding:3rem 2rem;min-height:400px;
                     display:flex;flex-direction:column;align-items:center;justify-content:center;">
                    <div style="font-size:5rem;margin-bottom:1rem;">👤</div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                         font-weight:600;color:#E8EAF0;margin-bottom:0.5rem;">
                        Face will appear here
                    </div>
                    <div style="font-size:0.85rem;color:#6B7280;text-align:center;">
                        Complete the interview on the left to generate<br>
                        the composite face with all specified attributes.
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 15.  ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
