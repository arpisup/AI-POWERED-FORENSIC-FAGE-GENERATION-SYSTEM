## 🔍 AI-Powered Forensic Face Generation System (v2.0)

This project is a **Streamlit-based application** that generates realistic human faces from witness descriptions using **Stable Diffusion v1.5**. It integrates **AI-driven prompt engineering, attribute modeling, and refinement techniques** to produce high-quality forensic facial composites.

### 🚀 Key Features

* **Confidence-Aware Prompting**

  * Each facial feature is weighted based on user confidence, improving realism and control.

* **Attribute Vector Integration (CelebA)**

  * Converts user inputs into a 40-dimensional attribute vector.
  * Injects attributes into prompts as weighted tokens for better generation accuracy.

* **Semantic Interpretation Engine**

  * Converts natural language descriptions into model-friendly tokens.

* **Image-to-Image Refinement**

  * Allows incremental edits while preserving identity (bone structure, lighting, etc.).

* **Multi-Level Generation Control**

  * Dynamically adjusts:

    * Guidance scale (faithfulness)
    * Inference steps (detail level)
    * Token weights (feature importance)

* **Before/After Comparison UI**

  * Visual comparison for iterative refinement.

* **Interpretability Dashboard**

  * Displays:

    * Prompt weights
    * Confidence metrics
    * Active attribute vector

* **Multiple Variations Generation**

  * Produces several face variations using controlled seed offsets.

---

### 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Stable Diffusion v1.5 (`diffusers`)
* **Libraries:** PyTorch, Transformers, PIL, NumPy

---

### ⚙️ How It Works

1. User provides facial descriptions step-by-step.
2. Each feature is assigned a **confidence score**.
3. Inputs are converted into:

   * Weighted text prompts
   * Attribute vectors
4. Stable Diffusion generates images using:

   * Text-to-image or image-to-image pipeline
5. User can refine specific features iteratively.

---

### 📦 Installation

```bash
pip install streamlit torch diffusers transformers accelerate pillow
streamlit run forensic_face_app_v2.py
```

---

### 🎯 Use Cases

* Criminal investigation support
* Missing person reconstruction
* Digital character prototyping
* AI-assisted portrait generation

---

### 📌 Highlights

* Combines **NLP + Computer Vision**
* Strong focus on **control, realism, and interpretability**
* Supports **iterative human-in-the-loop refinement**

---

