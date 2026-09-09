import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Page & Layout Settings ---
st.set_page_config(
    page_title="OptiVision Lab Studio | DIP Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Executive Dark Studio CSS Theme ---
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    /* Brand Banner */
    .studio-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0284C7 100%);
        padding: 24px 30px;
        border-radius: 14px;
        color: #F8FAFC;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
        border: 1px solid #334155;
    }
    .studio-banner h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .studio-banner p {
        font-size: 0.95rem;
        color: #BAE6FD;
        margin: 6px 0 0 0;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 16px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
    }
    /* Image Wrappers */
    div[data-testid="stImage"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="studio-banner">
    <h1>⚡ OptiVision Digital Image Processing Studio</h1>
    <p>Course: N-PECCS502P | Autonomous Academic Lab Portal | Department of Computer Science & Engineering</p>
</div>
""", unsafe_allow_html=True)

# --- Laboratory Syllabus & Viva Repository ---
LAB_METADATA = {
    "Practical 1: Python & OpenCV Setup": {
        "aim": "Prelab - Introduction to Python and Setup of Development Environment (necessary tools and libraries) for Image Processing (IDE PyCharm, OpenCV, NumPy, Matplotlib).",
        "objectives": [
            "Understand and run Python programs in different editors and IDEs.",
            "Install and import image processing libraries (OpenCV, NumPy, Matplotlib).",
            "Read, display, and inspect dimensions and metadata of images."
        ],
        "theory": """
**Image Array Geometry:**
- A digital image is represented as a 2D or 3D numeric matrix array in NumPy.
- **Grayscale:** Matrix $f(x, y)$ of height $H$ and width $W$, where pixel intensity $\in [0, 255]$.
- **RGB/BGR:** Tensor of shape $(H, W, 3)$ across Primary Color Channels. OpenCV reads color images in **BGR order** by default, whereas Matplotlib and PIL operate in **RGB order**.
        """,
        "viva": [
            ("**What is the role of NumPy in image processing?**", "OpenCV stores images as NumPy array matrices, enabling vectorized mathematical operations."),
            ("**Why does cv2.imread load images in BGR instead of RGB?**", "Historical conventions: early camera developers and frame grabbers formatted pixels as BGR in memory.")
        ]
    },
    "Practical 2: Color Formats, Arithmetic & Bitwise Logic": {
        "aim": "To convert images between various formats like RGB and Grayscale, perform arithmetic and bitwise operations on the images, and observe how these operations affect the image data representation.",
        "objectives": [
            "Understand the structure of RGB, Grayscale, and Binary formats.",
            "Perform arithmetic addition, weighted blending, and subtraction.",
            "Apply bitwise logic (AND, OR, NOT, XOR) for digital masking."
        ],
        "theory": """
**Arithmetic Operations:**
- **Weighted Blending:** $g(x, y) = \\alpha f_1(x, y) + \\beta f_2(x, y) + \\gamma$. Blends two signals while bounding intensities within $[0, 255]$.
- **Subtraction ($cv2.subtract$):** Computes differential intensity maps, useful for motion detection and change detection.

**Bitwise Boolean Operations:**
- **AND:** Extracts regions covered by a binary mask ($1 \\land 1 = 1$).
- **OR / XOR:** Merges features or isolates structural differences.
- **NOT:** Inverts pixel values ($255 - pixel$).
        """,
        "viva": [
            ("**Why is cv2.addWeighted used over direct addition?**", "Direct matrix addition wraps around modulo 256 in standard arithmetic or saturates heavily. cv2.addWeighted scales dynamically to avoid clipping."),
            ("**What information is lost during Grayscale conversion?**", "Chrominance (color hue and purity) is lost; only luminance (perceived brightness) is retained.")
        ]
    },
    "Practical 3: 2-D Geometric Transformations": {
        "aim": "Develop programs to apply 2-D geometric transformation operations: Translation, Rotation, Scaling, Shearing, Reflection, and Cropping.",
        "objectives": [
            "Implement basic 2-D transformations using affine transformation matrices.",
            "Apply reflection (flipping) and shearing along X and Y axes.",
            "Crop Regions of Interest (ROI) using coordinate array slicing."
        ],
        "theory": """
**Affine Geometry:**
All straight parallel lines remain parallel after mapping:
$$\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} a_{11} & a_{12} & t_x \\\\ a_{21} & a_{22} & t_y \\\\ 0 & 0 & 1 \\end{bmatrix} \\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}$$
- **Translation:** Linear displacement by offset $(t_x, t_y)$.
- **Rotation:** Rotation around center point $(\\text{cols}/2, \\text{rows}/2)$ at angle $\\theta$ via $cv2.getRotationMatrix2D$.
- **Shearing:** Displaces one coordinate proportionally based on the orthogonal axis.
        """,
        "viva": [
            ("**What is the difference between affine and perspective transforms?**", "Affine uses a $2 \\times 3$ matrix preserving parallelism; perspective (homography) uses a $3 \\times 3$ matrix for arbitrary viewing projections."),
            ("**Why do we use cv2.INTER_CUBIC during enlargement?**", "Bicubic interpolation samples a $4 \\times 4$ neighborhood, generating smoother gradients than nearest-neighbor or bilinear methods.")
        ]
    },
    "Practical 4: Spatial Enhancements & Equalization": {
        "aim": "Study and implement spatial domain image enhancement techniques: Histogram Equalization, Brightness/Contrast adjustment, and Thresholding.",
        "objectives": [
            "Improve global dynamic contrast using Histogram Equalization.",
            "Tune intensity dynamic range using linear scaling parameters ($\alpha$ and $\beta$).",
            "Segment images using simple, inverted, truncated, and to-zero thresholding."
        ],
        "theory": """
**Spatial Domain Transfer Functions:**
- **Linear Scaling:** $g(x, y) = \\alpha \\cdot f(x, y) + \\beta$, where $\\alpha$ scales slope (contrast) and $\\beta$ shifts intercept (brightness).
- **Histogram Equalization ($cv2.equalizeHist$):** Flattens the cumulative distribution function (CDF) across $[0, 255]$ to maximize contrast across low-contrast images.
- **Thresholding:** Maps continuous tones into binary foreground/background masks based on scalar cutoff $T$.
        """,
        "viva": [
            ("**How does Histogram Equalization enhance contrast?**", "It reassigns pixel values so that frequent intensity levels occupy a broader range in the histogram."),
            ("**When does Histogram Equalization fail?**", "When the background contains high-contrast noise, equalization will amplify the background noise unnaturally.")
        ]
    },
    "Practical 5: Spatial Filtering & Smoothing": {
        "aim": "Write Python programs using OpenCV to apply spatial domain filters: Averaging, Gaussian, Median, and Bilateral filters.",
        "objectives": [
            "Understand spatial 2D convolution using neighborhood masks.",
            "Reduce noise using low-pass linear smoothing filters.",
            "Compare non-linear median filtering with edge-preserving bilateral filtering."
        ],
        "theory": """
**Spatial Filtering Masks:**
- **Averaging Filter ($cv2.blur$):** Uniform box kernel taking the arithmetic mean of all pixels under the window.
- **Gaussian Filter:** 2D bell curve kernel placing highest weight at the center pixel, smoothing high-frequency noise.
- **Median Filter:** Order-statistic filter replacing the center pixel with the neighborhood median; suppresses impulsive salt-and-pepper noise.
- **Bilateral Filter:** Dual-Gaussian kernel operating in coordinate space and radiometric intensity space, smoothing texture while preserving edges.
        """,
        "viva": [
            ("**Why is Bilateral filtering favored for facial skin smoothing?**", "It blurs subtle blemishes in flat skin areas while preventing edge degradation around eyes and lips."),
            ("**Why does Median filtering preserve edges better than Averaging?**", "Because the median is an order-statistic selector, not an arithmetic averager; it does not generate intermediate blurry shades.")
        ]
    },
    "Practical 6: Image Restoration & Inpainting": {
        "aim": "Implement image inpainting using the Fast Marching (Telea) and Navier-Stokes (NS) methods to restore corrupted areas.",
        "objectives": [
            "Study degradation models and restoration fundamentals.",
            "Construct binary masks highlighting damaged pixel regions.",
            "Compare Telea Fast Marching and Navier-Stokes inpainting algorithms."
        ],
        "theory": """
**Image Inpainting Principles:**
Reconstructs lost, scratched, or corrupted regions using neighboring valid pixels:
- **Telea Method (Fast Marching):** Advances inward from boundary pixels, computing weighted averages based on distance and gradient vectors.
- **Navier-Stokes Method:** Derives from computational fluid dynamics, propagating isophotes (contours of constant intensity) into missing areas.
        """,
        "viva": [
            ("**What is an inpainting mask?**", "A binary 8-bit mask where white pixels ($255$) indicate damaged regions to fill, and black pixels ($0$) represent original content."),
            ("**Which method is faster for thin scratch removal?**", "The Telea method executes considerably faster and produces clean results on thin lines and scratches.")
        ]
    },
    "Practical 7: Lossless vs Lossy Compression": {
        "aim": "Implement image coding techniques to achieve compression and compare original and compressed file sizes.",
        "objectives": [
            "Examine spatial, coding, and psychovisual redundancies.",
            "Implement lossy JPEG compression across quality factors.",
            "Evaluate lossless PNG compression and calculate compression ratios."
        ],
        "theory": """
**Compression Metrics:**
- **Compression Ratio ($CR$):** $CR = \\frac{\\text{Raw Size}}{\\text{Compressed Size}}$.
- **Lossy Compression (JPEG):** Converts blocks via Discrete Cosine Transform (DCT) and applies scalar quantization, discarding subtle color details.
- **Lossless Compression (PNG):** Employs 2D predictive filters followed by Deflate (LZ77 + Huffman coding) for bit-exact recovery.
        """,
        "viva": [
            ("**What is psychovisual redundancy?**", "Visual information that the human visual system (HVS) cannot perceive, which can be discarded without visible distortion."),
            ("**Why does raw uncompressed BMP take so much space?**", "BMP stores every pixel channel individually without encoding repeated adjacent pixel values.")
        ]
    },
    "Practical 8: Morphological Operations": {
        "aim": "Perform morphological operations: Erosion, Dilation, Opening, and Closing on binary images to analyze object boundaries and noise removal.",
        "objectives": [
            "Implement erosion and dilation using structuring elements (kernels).",
            "Apply opening and closing for structural smoothing and hole filling.",
            "Extract object contours and compute morphological component areas."
        ],
        "theory": """
**Set-Theoretic Morphology:**
- **Erosion ($A \\ominus B$):** Shrinks foreground boundaries where the structuring element completely fits.
- **Dilation ($A \\oplus B$):** Expands foreground regions where the structuring element hits the object.
- **Opening ($A \\circ B$):** Erosion followed by Dilation. Eliminates small bright noise specks.
- **Closing ($A \\bullet B$):** Dilation followed by Erosion. Bridges narrow gaps and fills small holes.
        """,
        "viva": [
            ("**What is a structuring element?**", "A small matrix mask (rectangle, cross, or ellipse) defining the neighborhood tested during morphological traversal."),
            ("**How does morphological gradient find boundaries?**", "By taking the pixel difference between Dilation and Erosion: $G = (A \\oplus B) - (A \\ominus B)$.")
        ]
    },
    "Practical 9: Correlation-based Object Detection": {
        "aim": "Develop an object detection program using the Correlation Principle (Template Matching).",
        "objectives": [
            "Locate specific objects by sliding a template patch across an image.",
            "Compute Normalized Cross-Correlation (NCC) metrics.",
            "Threshold correlation response maps and draw bounding boxes."
        ],
        "theory": """
**Normalized Cross-Correlation (NCC):**
Slides template $T$ across source image $I$:
$$R(x, y) = \\frac{\\sum_{x', y'} (T(x', y') \\cdot I(x+x', y+y'))}{\\sqrt{\\sum T(x', y')^2 \\cdot \\sum I(x+x', y+y')^2}}$$
OpenCV's `cv2.TM_CCOEFF_NORMED` yields a score from $-1.0$ (complete mismatch) to $+1.0$ (perfect match).
        """,
        "viva": [
            ("**Why does template matching fail when an object is rotated?**", "NCC is not rotation-invariant; changing object orientation reduces pixel correlation significantly."),
            ("**What is the output format of cv2.matchTemplate?**", "A 2D single-channel float matrix of dimensions $(W - w + 1) \\times (H - h + 1)$.")
        ]
    },
    "Post-Lab 2: Advanced Color Spaces & Channels": {
        "aim": "Convert images between the RGB, HSV, YCrCb, and Lab colour spaces, analyzing how colour information is encoded in each.",
        "objectives": [
            "Transform images between RGB, HSV, YCrCb, and CIELAB spaces.",
            "Analyze hue, saturation, chrominance, and luminance decoupling.",
            "Inspect individual single-channel intensity maps."
        ],
        "theory": """
**Color Spaces in Computer Vision:**
- **RGB:** Standard display format; luminance and chrominance are coupled across all 3 channels.
- **HSV:** Separates chromatic data (Hue $0-179$, Saturation $0-255$) from Luminance (Value $0-255$).
- **YCrCb:** Separates Luminance ($Y$) from Chroma differences ($Cr, Cb$), enabling JPEG chroma subsampling.
- **CIELAB:** Perceptually uniform space based on human color reception ($L^*$ Lightness, $a^*$ green-red, $b^*$ blue-yellow).
        """,
        "viva": [
            ("**Why is HSV ideal for skin or object color tracking?**", "Hue remains constant under variable shadows and brightness levels."),
            ("**What is CIELAB perceptual uniformity?**", "A unit change in Euclidean distance between two Lab coordinates produces an equal perceived visual difference.")
        ]
    },
    "Post-Lab 3: Edge Detection (Canny, Sobel, Prewitt)": {
        "aim": "Detect edges in images with the Canny method and contrast the results with Sobel and Prewitt detectors.",
        "objectives": [
            "Implement multi-stage Canny edge detection on grayscale images.",
            "Apply Sobel and Prewitt gradient convolution kernels.",
            "Contrast results regarding edge continuity, thickness, and noise sensitivity."
        ],
        "theory": """
**Differential Edge Detectors:**
1. **Sobel:** Uses weighted kernels emphasizing center pixels:
   $$K_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -2 & 0 & 2 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad K_y = \\begin{bmatrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 2 & 1 \\end{bmatrix}$$
2. **Prewitt:** Uses uniform gradient masks without central weighting:
   $$K_x = \\begin{bmatrix} -1 & 0 & 1 \\\\ -1 & 0 & 1 \\\\ -1 & 0 & 1 \\end{bmatrix}, \\quad K_y = \\begin{bmatrix} -1 & -1 & -1 \\\\ 0 & 0 & 0 \\\\ 1 & 1 & 1 \\end{bmatrix}$$
3. **Canny Multi-Stage Algorithm:**
   - Gaussian Blur $\\rightarrow$ Gradient Computation $\\rightarrow$ Non-Maximum Suppression (thins edges to 1 pixel) $\\rightarrow$ Hysteresis Double Thresholding.
        """,
        "viva": [
            ("**Why does Canny produce clean 1-pixel edges while Sobel produces thick bands?**", "Canny's non-maximum suppression step discards all gradient values that are not local directional maxima."),
            ("**What is hysteresis thresholding in Canny?**", "Pixels above $T_{\\text{high}}$ are retained as strong edges; pixels between $T_{\\text{low}}$ and $T_{\\text{high}}$ are kept only if connected to strong edges.")
        ]
    }
}

# --- Preset Benchmark Generators (Zero-friction Testing) ---
@st.cache_data
def generate_preset(name):
    if name == "Synthetic Shapes":
        img = np.ones((400, 400, 3), dtype=np.uint8) * 45
        cv2.circle(img, (140, 140), 75, (230, 80, 50), -1)
        cv2.rectangle(img, (220, 200), (360, 340), (40, 200, 120), -1)
        cv2.putText(img, "DIP LAB", (80, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        return img
    elif name == "High-Contrast Pattern":
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(0, 400, 40):
            cv2.line(img, (i, 0), (i, 400), (180, 180, 180), 2)
            cv2.line(img, (0, i), (400, i), (180, 180, 180), 2)
        cv2.circle(img, (200, 200), 100, (255, 215, 0), -1)
        return img
    elif name == "Text & Document":
        img = np.ones((400, 400, 3), dtype=np.uint8) * 240
        cv2.putText(img, "CONFIDENTIAL", (40, 90), cv2.FONT_HERSHEY_DUPLEX, 1.1, (20, 20, 20), 2)
        cv2.putText(img, "Image Processing Lab", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        cv2.putText(img, "Department of CSE", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        cv2.line(img, (40, 230), (360, 230), (0, 0, 180), 3)
        return img
    else: # Gradient Spectrum
        x = np.linspace(0, 255, 400, dtype=np.uint8)
        grid = np.tile(x, (400, 1))
        return cv2.applyColorMap(grid, cv2.COLORMAP_JET)

# --- Quantitative Quality Metrics ---
def compute_psnr_mse(orig, proc):
    if len(orig.shape) == 3 and len(proc.shape) == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
    elif len(orig.shape) == 2 and len(proc.shape) == 3:
        proc = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
    
    if orig.shape != proc.shape:
        proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
        
    mse = np.mean((orig.astype(np.float64) - proc.astype(np.float64)) ** 2)
    if mse == 0:
        return 0.0, 100.0
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr

def get_png_bytes(img_array):
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### 🎛️ Navigation & Experiment Select")
    selected_exp = st.selectbox("Active Practical", list(LAB_METADATA.keys()))
    st.markdown("---")
    
    st.markdown("### 🖼️ Input Image Source")
    input_source = st.radio("Choose Input Mode", ["Built-in Benchmarks (Fast)", "Upload Local File", "Live Camera Snap"])
    
    img_rgb = None
    if input_source == "Built-in Benchmarks (Fast)":
        preset = st.selectbox("Select Preset Image", ["Synthetic Shapes", "High-Contrast Pattern", "Text & Document", "Gradient Spectrum"])
        img_rgb = generate_preset(preset)
    elif input_source == "Upload Local File":
        up_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up_file is not None:
            img_rgb = np.array(Image.open(up_file).convert("RGB"))
    else:
        cam_shot = st.camera_input("Snap photo via webcam")
        if cam_shot is not None:
            img_rgb = np.array(Image.open(cam_shot).convert("RGB"))

meta = LAB_METADATA[selected_exp]

# --- Main Tabs ---
tab_app, tab_theory, tab_viva = st.tabs(["🚀 Interactive Workbench", "🎯 Aim, Objective & Theory", "📝 Viva Voce & Discussion"])

# -------------------------------------------------------------
# TAB 2: AIM, OBJECTIVES & THEORY
# -------------------------------------------------------------
with tab_theory:
    st.subheader(f"📌 {selected_exp}")
    st.markdown(f"**Aim:** {meta['aim']}")
    st.markdown("---")
    st.markdown("#### 🎯 Expected Learning Outcomes")
    for obj in meta["objectives"]:
        st.markdown(f"- {obj}")
    st.markdown("---")
    st.markdown("#### 📖 Mathematical Formulation & Principles")
    st.markdown(meta["theory"])

# -------------------------------------------------------------
# TAB 3: VIVA VOCE & DISCUSSION
# -------------------------------------------------------------
with tab_viva:
    st.subheader("💡 Lab Discussion & Oral Examination Prep")
    for q, a in meta["viva"]:
        with st.expander(q):
            st.write(a)

# -------------------------------------------------------------
# TAB 1: INTERACTIVE WORKBENCH
# -------------------------------------------------------------
with tab_app:
    if img_rgb is None:
        st.info("👈 Please select a benchmark preset or upload an image from the sidebar to launch the workbench.")
    else:
        rows, cols = img_rgb.shape[:2]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Metric Header Ribbon
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Resolution", f"{cols} × {rows} px")
        col_m2.metric("Color Channels", f"{img_rgb.shape[2]}")
        col_m3.metric("Mean Luminance", f"{img_gray.mean():.1f}")
        col_m4.metric("Dynamic Range", f"[{img_gray.min()}, {img_gray.max()}]")

        st.markdown("---")

        # PRACTICAL 1
        if "Practical 1" in selected_exp:
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Source Input", use_container_width=True)
            with c2:
                mode = st.selectbox("Inspect Matrix Channel", ["Grayscale", "Red Intensity Plane", "Green Intensity Plane", "Blue Intensity Plane"])
                if mode == "Grayscale":
                    res = img_gray
                elif mode == "Red Intensity Plane":
                    res = img_rgb[:, :, 0]
                elif mode == "Green Intensity Plane":
                    res = img_rgb[:, :, 1]
                else:
                    res = img_rgb[:, :, 2]
                st.image(res, caption=f"View: {mode}", use_container_width=True)
                st.download_button("📥 Export Matrix View", get_png_bytes(res), "channel.png")

        # PRACTICAL 2
        elif "Practical 2" in selected_exp:
            op_group = st.radio("Select Operation Group", ["Arithmetic Blending & Differential", "Bitwise Boolean Gates"], horizontal=True)
            if op_group == "Arithmetic Blending & Differential":
                st.write("**Secondary Signal Input:**")
                sec_mode = st.radio("Secondary Image", ["Synthetic Complement", "Upload Image"], horizontal=True)
                if sec_mode == "Synthetic Complement":
                    img2_res = cv2.bitwise_not(img_rgb)
                else:
                    f2 = st.file_uploader("Upload Second Image", type=["jpg", "png"], key="p2_f2")
                    img2_res = cv2.resize(np.array(Image.open(f2).convert("RGB")), (cols, rows)) if f2 else None

                if img2_res is not None:
                    alpha = st.slider("Primary Image Weight (Alpha)", 0.0, 1.0, 0.6)
                    blended = cv2.addWeighted(img_rgb, alpha, img2_res, 1.0 - alpha, 0)
                    subtracted = cv2.subtract(img_rgb, img2_res)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.image(img_rgb, caption="Primary Image", use_container_width=True)
                    c2.image(blended, caption=f"Weighted Blend (Alpha={alpha:.2f})", use_container_width=True)
                    c3.image(subtracted, caption="Direct Subtraction Map", use_container_width=True)
            else:
                gate = st.selectbox("Select Gate", ["Bitwise NOT", "Bitwise AND (Circular Aperture Mask)", "Bitwise OR", "Bitwise XOR"])
                mask = np.zeros((rows, cols), dtype=np.uint8)
                cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 3, 255, -1)
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Original Signal", use_container_width=True)
                if gate == "Bitwise NOT":
                    res = cv2.bitwise_not(img_rgb)
                elif gate == "Bitwise AND (Circular Aperture Mask)":
                    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                elif gate == "Bitwise OR":
                    res = cv2.bitwise_or(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                else:
                    res = cv2.bitwise_xor(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                c2.image(res, caption=gate, use_container_width=True)

        # PRACTICAL 3
        elif "Practical 3" in selected_exp:
            trans = st.selectbox("Transformation Type", ["Translation", "Rotation", "Scaling", "Reflection", "Shearing", "Cropping (ROI)"])
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Base Coordinate Frame", use_container_width=True)
            with c2:
                if trans == "Translation":
                    tx = st.slider("X-Displacement", -200, 200, 60)
                    ty = st.slider("Y-Displacement", -200, 200, 40)
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    res = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif trans == "Rotation":
                    deg = st.slider("Rotation Angle (Degrees)", -180, 180, 35)
                    scale = st.slider("Scale Factor", 0.4, 1.5, 0.9)
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, scale)
                    res = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif trans == "Scaling":
                    fx = st.slider("Scale X", 0.3, 2.0, 1.2)
                    fy = st.slider("Scale Y", 0.3, 2.0, 0.8)
                    res = cv2.resize(img_rgb, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                elif trans == "Reflection":
                    axis = st.radio("Axis", ["Horizontal Flip", "Vertical Flip"])
                    res = cv2.flip(img_rgb, 1 if "Horizontal" in axis else 0)
                elif trans == "Shearing":
                    sh_axis = st.radio("Shear Plane", ["X-Axis", "Y-Axis"])
                    sh_val = st.slider("Shear Factor", 0.1, 0.8, 0.35)
                    if sh_axis == "X-Axis":
                        M = np.float32([[1, sh_val, 0], [0, 1, 0], [0, 0, 1]])
                    else:
                        M = np.float32([[1, 0, 0], [sh_val, 1, 0], [0, 0, 1]])
                    res = cv2.warpPerspective(img_rgb, M, (int(cols * 1.4), int(rows * 1.4)))
                else: # Cropping
                    y1 = st.slider("Start Row (Y1)", 0, rows // 2, 20)
                    y2 = st.slider("End Row (Y2)", rows // 2, rows, rows - 20)
                    x1 = st.slider("Start Col (X1)", 0, cols // 2, 20)
                    x2 = st.slider("End Col (X2)", cols // 2, cols, cols - 20)
                    res = img_rgb[y1:y2, x1:x2]
                st.image(res, caption=f"Transformed: {trans}", use_container_width=True)

        # PRACTICAL 4
        elif "Practical 4" in selected_exp:
            enh = st.selectbox("Enhancement Strategy", ["Histogram Equalization & CDF", "Brightness & Contrast Transfer", "Thresholding Models"])
            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Original Grayscale", use_container_width=True)
            with c2:
                if enh == "Histogram Equalization & CDF":
                    res = cv2.equalizeHist(img_gray)
                    st.image(res, caption="Equalized Tone Mapping", use_container_width=True)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
                    ax1.hist(img_gray.ravel(), 256, [0, 256], color='#38BDF8')
                    ax1.set_title("Input Histogram")
                    ax2.hist(res.ravel(), 256, [0, 256], color='#10B981')
                    ax2.set_title("Equalized Histogram")
                    st.pyplot(fig)
                elif enh == "Brightness & Contrast Transfer":
                    a = st.slider("Contrast Alpha (Gain)", 0.5, 3.0, 1.5)
                    b = st.slider("Brightness Beta (Bias)", -100, 100, 25)
                    res = cv2.convertScaleAbs(img_gray, alpha=a, beta=b)
                    st.image(res, caption=f"Scaled (Alpha={a}, Beta={b})", use_container_width=True)
                else:
                    th = st.slider("Threshold Cutoff Level", 0, 255, 128)
                    t_type = st.selectbox("Threshold Technique", ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_OTSU"])
                    if t_type == "THRESH_OTSU":
                        ret, res = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        st.write(f"**Otsu Computed Optimal Threshold:** `{ret}`")
                    else:
                        _, res = cv2.threshold(img_gray, th, 255, getattr(cv2, t_type))
                    st.image(res, caption=f"Result ({t_type})", use_container_width=True)

        # PRACTICAL 5
        elif "Practical 5" in selected_exp:
            f_type = st.selectbox("Filter Class", ["Averaging Box Filter", "Gaussian Bell Curve", "Median Filter", "Bilateral Filter"])
            k = st.slider("Window Kernel Size (Odd)", 3, 25, 5, step=2)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Raw Spatial Domain", use_container_width=True)
            with c2:
                if f_type == "Averaging Box Filter":
                    res = cv2.blur(img_rgb, (k, k))
                elif f_type == "Gaussian Bell Curve":
                    sigma = st.slider("Gaussian Sigma", 0.2, 10.0, 1.5)
                    res = cv2.GaussianBlur(img_rgb, (k, k), sigma)
                elif f_type == "Median Filter":
                    res = cv2.medianBlur(img_rgb, k)
                else:
                    res = cv2.bilateralFilter(img_rgb, k, 75, 75)
                st.image(res, caption=f"Smoothed Output ({f_type})", use_container_width=True)
                
                # Statistical Signal Quality
                mse_val, psnr_val = compute_psnr_mse(img_rgb, res)
                st.caption(f"📊 **Objective Quality Metrics:** MSE = `{mse_val:.2f}` | PSNR = `{psnr_val:.2f} dB`")

        # PRACTICAL 6
        elif "Practical 6" in selected_exp:
            st.write("Artificial degradation and inpainting restoration:")
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                scratch_thick = st.slider("Scratch Thickness", 2, 8, 3)
            with col_d2:
                inpaint_rad = st.slider("Inpainting Neighborhood Radius", 1, 10, 3)

            # Generate synthetic degradation mask
            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.line(mask, (cols // 5, rows // 4), (4 * cols // 5, rows // 4), 255, scratch_thick)
            cv2.line(mask, (cols // 2, rows // 6), (cols // 2, 5 * rows // 6), 255, scratch_thick)
            damaged = img_rgb.copy()
            damaged[mask == 255] = [255, 255, 255]

            alg = st.radio("Inpainting Algorithm", ["Telea Fast Marching (cv2.INPAINT_TELEA)", "Navier-Stokes (cv2.INPAINT_NS)"], horizontal=True)
            flag = cv2.INPAINT_TELEA if "Telea" in alg else cv2.INPAINT_NS
            restored = cv2.inpaint(damaged, mask, inpaint_rad, flag)

            c1, c2, c3 = st.columns(3)
            c1.image(damaged, caption="Degraded Image", use_container_width=True)
            c2.image(mask, caption="Damage Mask (255)", use_container_width=True)
            c3.image(restored, caption=f"Restored ({alg.split()[0]})", use_container_width=True)
            
            _, psnr_res = compute_psnr_mse(img_rgb, restored)
            st.caption(f"✨ **Reconstruction Fidelity to Ground Truth:** PSNR = `{psnr_res:.2f} dB`")

        # PRACTICAL 7
        elif "Practical 7" in selected_exp:
            q = st.slider("JPEG Quality Factor (Lossy)", 5, 100, 25)
            png_lvl = st.slider("PNG Deflate Compression Level (Lossless)", 0, 9, 9)
            
            _, enc_jpg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            _, enc_png = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_lvl])
            
            raw_kb = (rows * cols * 3) / 1024
            jpg_kb = len(enc_jpg) / 1024
            png_kb = len(enc_png) / 1024
            
            st.table({
                "Algorithm Format": ["Raw Uncompressed", f"Lossy JPEG (Quality={q})", f"Lossless PNG (Level={png_lvl})"],
                "File Size (KB)": [f"{raw_kb:.1f} KB", f"{jpg_kb:.1f} KB", f"{png_kb:.1f} KB"],
                "Compression Ratio": ["1.00:1", f"{(raw_kb/jpg_kb):.2f}:1", f"{(raw_kb/png_kb):.2f}:1"],
                "Data Integrity": ["100% Bit-Exact", "Lossy (Psychovisual Quantized)", "100% Exact Reconstruction"]
            })
            
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(cv2.imdecode(enc_jpg, 1), cv2.COLOR_BGR2RGB), caption=f"JPEG Output ({jpg_kb:.1f} KB)", use_container_width=True)
            c2.image(cv2.cvtColor(cv2.imdecode(enc_png, 1), cv2.COLOR_BGR2RGB), caption=f"PNG Output ({png_kb:.1f} KB)", use_container_width=True)

        # PRACTICAL 8
        elif "Practical 8" in selected_exp:
            op = st.selectbox("Morphological Operator", ["Erosion", "Dilation", "Opening (Noise Cleanse)", "Closing (Hole Fill)", "Morphological Gradient"])
            k_sz = st.slider("Structuring Element Size", 3, 19, 5, step=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
            
            _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            ops = {
                "Erosion": cv2.erode(binary, kernel),
                "Dilation": cv2.dilate(binary, kernel),
                "Opening (Noise Cleanse)": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                "Closing (Hole Fill)": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
                "Morphological Gradient": cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            }
            res = ops[op]
            
            # Contour analysis
            contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contoured_canvas = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(contoured_canvas, contours, -1, (255, 0, 0), 2)
            
            c1, c2, c3 = st.columns(3)
            c1.image(binary, caption="Binary Base", use_container_width=True)
            c2.image(res, caption=f"Morphology: {op}", use_container_width=True)
            c3.image(contoured_canvas, caption=f"Contours Detected: {len(contours)}", use_container_width=True)

        # PRACTICAL 9
        elif "Practical 9" in selected_exp:
            st.write("Dynamic template extraction and Normalized Cross-Correlation:")
            tw = st.slider("Template Width", 40, 160, 80)
            th = st.slider("Template Height", 40, 160, 80)
            
            cy, cx = rows // 2, cols // 2
            template = img_gray[cy:cy + th, cx:cx + tw]
            
            corr = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = st.slider("Match Correlation Threshold", 0.5, 0.99, 0.8)
            loc = np.where(corr >= threshold)
            
            detected = img_rgb.copy()
            match_count = 0
            for pt in zip(*loc[::-1]):
                cv2.rectangle(detected, pt, (pt[0] + tw, pt[1] + th), (255, 230, 0), 2)
                match_count += 1
                
            c1, c2 = st.columns([1, 3])
            c1.image(template, caption="Template Sub-Window", use_container_width=True)
            c2.image(detected, caption=f"Detections Found ({match_count} regions)", use_container_width=True)

        # POST-LAB 2
        elif "Post-Lab 2" in selected_exp:
            space = st.selectbox("Color Space Decompositions", ["HSV (Hue/Sat/Val)", "YCrCb (Broadcast/Luma)", "CIELAB (Uniform)"])
            if space == "HSV (Hue/Sat/Val)":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                labels = ["Hue (H)", "Saturation (S)", "Value (V)"]
            elif space == "YCrCb (Broadcast/Luma)":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                labels = ["Luminance (Y)", "Chroma Red (Cr)", "Chroma Blue (Cb)"]
            else:
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                labels = ["Lightness (L)", "Green-Red (a*)", "Blue-Yellow (b*)"]
                
            c1, c2, c3 = cv2.split(converted)
            st.image(converted, caption=f"Full {space} Mapping", use_container_width=True)
            ca, cb, cc = st.columns(3)
            ca.image(c1, caption=labels[0], use_container_width=True)
            cb.image(c2, caption=labels[1], use_container_width=True)
            cc.image(c3, caption=labels[2], use_container_width=True)

        # POST-LAB 3
        else:
            edge_view = st.selectbox("Edge Detection Mode", [
                "Side-by-Side Comparison (Canny vs Sobel vs Prewitt)",
                "Canny Tunable Multi-Stage Engine",
                "Sobel Operator (Directional Magnitudes)",
                "Prewitt Operator (Convolution Masks)"
            ])

            if edge_view == "Side-by-Side Comparison (Canny vs Sobel vs Prewitt)":
                c_edge = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 1.4), 50, 150)

                sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
                sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
                s_edge = np.uint8(np.clip(cv2.magnitude(sx, sy), 0, 255))

                px = cv2.filter2D(img_gray, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32))
                py = cv2.filter2D(img_gray, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32))
                p_edge = np.uint8(np.clip(cv2.magnitude(px, py), 0, 255))

                c1, c2, c3 = st.columns(3)
                c1.image(c_edge, caption="Canny (1-px Non-Max Suppressed)", use_container_width=True)
                c2.image(s_edge, caption="Sobel (First-Order Gradient)", use_container_width=True)
                c3.image(p_edge, caption="Prewitt (Uniform Spatial Convolution)", use_container_width=True)
            elif edge_view == "Canny Tunable Multi-Stage Engine":
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    t_low = st.slider("Hysteresis Lower Threshold", 0, 255, 50)
                    t_high = st.slider("Hysteresis Upper Threshold", 0, 255, 150)
                with col_t2:
                    aperture = st.selectbox("Sobel Aperture Size", [3, 5, 7], index=0)
                    use_l2 = st.checkbox("Enable L2 Euclidean Norm", value=True)
                
                canny_res = cv2.Canny(cv2.GaussianBlur(img_gray, (5, 5), 1.4), t_low, t_high, apertureSize=aperture, L2gradient=use_l2)
                c1, c2 = st.columns(2)
                c1.image(img_gray, caption="Monochrome Base", use_container_width=True)
                c2.image(canny_res, caption=f"Canny Output (Lower={t_low}, Upper={t_high})", use_container_width=True)
            elif edge_view == "Sobel Operator (Directional Magnitudes)":
                ksize = st.selectbox("Kernel Size", [3, 5, 7], index=0)
                sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
                sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
                s_mag = np.uint8(np.clip(cv2.magnitude(sx, sy), 0, 255))
                c1, c2, c3 = st.columns(3)
                c1.image(np.uint8(np.absolute(sx)), caption="Sobel X (Vertical Edges)", use_container_width=True)
                c2.image(np.uint8(np.absolute(sy)), caption="Sobel Y (Horizontal Edges)", use_container_width=True)
                c3.image(s_mag, caption="Sobel Magnitude", use_container_width=True)
            else:
                kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
                ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
                px = cv2.filter2D(img_gray, cv2.CV_64F, kx)
                py = cv2.filter2D(img_gray, cv2.CV_64F, ky)
                p_mag = np.uint8(np.clip(cv2.magnitude(px, py), 0, 255))
                c1, c2, c3 = st.columns(3)
                c1.image(np.uint8(np.absolute(px)), caption="Prewitt X", use_container_width=True)
                c2.image(np.uint8(np.absolute(py)), caption="Prewitt Y", use_container_width=True)
                c3.image(p_mag, caption="Prewitt Magnitude", use_container_width=True)
