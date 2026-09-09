import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="VisionCraft | DIP Interactive Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 22px 26px;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .main-header h1 {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        color: #38BDF8;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 4px 0 0 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔬 Digital Image Processing Lab Workbench</h1>
    <p>Course: N-PECCS502P | S. B. Jain Institute of Technology, Management & Research, Nagpur</p>
</div>
""", unsafe_allow_html=True)

# --- Experiments Registry (Aim, Objective, Theory, Viva) ---
LAB_METADATA = {
    "Practical 1: Python & OpenCV Setup": {
        "aim": "Prelab - Introduction to Python and Setup of Development Environment (necessary tools and libraries) for Image Processing (IDE PyCharm, OpenCV, NumPy, Matplotlib).",
        "objectives": [
            "Understand and run Python programs in different editors and IDEs.",
            "Install and import image processing libraries (OpenCV, NumPy, Matplotlib).",
            "Read, display, and inspect dimensions and metadata of images."
        ],
        "theory": """
**Python & Image Processing Libraries:**
- **OpenCV:** Open-Source Computer Vision Library providing cross-platform computer vision and image processing functions.
- **NumPy:** Fundamental package for multidimensional arrays; digital images are represented as numerical matrix arrays.
- **Matplotlib:** Visualization package used for plotting channel maps and histograms.
- **Image Representation:** Grayscale images are 2D arrays $f(x, y)$; color images are 3D arrays of size $(H \times W \times C)$.
        """,
        "viva": [
            ("**What is the role of NumPy in image processing?**", "OpenCV stores and manipulates digital images as NumPy array matrices, enabling fast vector operations."),
            ("**How does OpenCV store color channels compared to Matplotlib?**", "OpenCV reads images in BGR format by default, whereas Matplotlib uses RGB.")
        ]
    },
    "Practical 2: Color Formats, Arithmetic & Bitwise Logic": {
        "aim": "To convert images between various formats like RGB and Grayscale, perform arithmetic and bitwise operations on the images, and observe how these operations affect the image data representation.",
        "objectives": [
            "Understand the structure of RGB, Grayscale, and Binary image formats.",
            "Perform arithmetic operations (Addition, Subtraction) and analyze saturation effects.",
            "Apply bitwise logic (AND, OR, NOT, XOR) for masking and feature isolation."
        ],
        "theory": """
**Arithmetic Operations:**
- **Addition ($cv2.addWeighted$):** Blends two images using weights: $g(x,y) = \alpha f_1(x,y) + \beta f_2(x,y) + \gamma$.
- **Subtraction ($cv2.subtract$):** Computes pixel-wise difference, highlighting differences or changes between two images.

**Bitwise Operations:**
- **Bitwise AND:** Useful for masking and regional extraction.
- **Bitwise OR / XOR:** Combines image components or segments discrepancies.
- **Bitwise NOT:** Inverts image intensities ($255 - pixel$).
        """,
        "viva": [
            ("**Why is cv2.addWeighted preferred over direct pixel addition?**", "Direct addition can cause saturation clipping at 255 (producing solid white spots). Weighted addition smoothly balances brightness."),
            ("**What information is lost when converting RGB to Grayscale?**", "Chrominance (color hue and saturation) is discarded; only luminance (brightness) is preserved.")
        ]
    },
    "Practical 3: 2-D Geometric Transformations": {
        "aim": "Develop programs to apply 2-D geometric transformation operations: Translation, Rotation, Scaling, Shearing, Reflection, and Cropping.",
        "objectives": [
            "Implement basic 2-D transformations using affine matrices.",
            "Apply reflection (flipping) and shearing along X and Y axes.",
            "Crop Regions of Interest (ROI) using coordinate slicing."
        ],
        "theory": """
**Affine Transformations:**
All parallel lines in the original image remain parallel after transformation.
- **Translation:** Shift by $[t_x, t_y]$ using matrix $\\begin{bmatrix} 1 & 0 & t_x \\\\ 0 & 1 & t_y \\end{bmatrix}$.
- **Rotation:** Rotates around center point $(\\text{cols}/2, \\text{rows}/2)$ at angle $\\theta$ via $cv2.getRotationMatrix2D$.
- **Shearing:** Displaces coordinates proportionally along one axis while fixing the other.
- **Reflection:** Flips along horizontal ($Sy = -1$) or vertical ($Sx = -1$) axes.
        """,
        "viva": [
            ("**What is the role of a transformation matrix?**", "It maps coordinate points $(x, y)$ in the input image to destination coordinates $(x', y')$."),
            ("**What is the difference between affine and perspective transformation?**", "Affine uses a 2x3 matrix preserving parallelism; perspective uses a 3x3 matrix for non-parallel projections.")
        ]
    },
    "Practical 4: Spatial Enhancements & Equalization": {
        "aim": "Study and implement spatial domain image enhancement techniques: Histogram Equalization, Brightness/Contrast adjustment, and Thresholding.",
        "objectives": [
            "Improve global contrast using Histogram Equalization.",
            "Tune dynamic range using linear scaling parameters (alpha and beta).",
            "Segment images using simple, inverted, truncated, and to-zero thresholding."
        ],
        "theory": """
**Spatial Domain Processing:**
Manipulates pixel intensities directly: $g(x, y) = T[f(x, y)]$.
- **Linear Scaling:** $g(x, y) = \\alpha \\cdot f(x, y) + \\beta$, where $\\alpha$ controls contrast and $\\beta$ controls brightness.
- **Histogram Equalization:** Redistributes frequent intensity levels across the entire range [0, 255] to enhance low-contrast images.
- **Thresholding:** Converts grayscale images into binary foreground/background masks based on a cutoff $T$.
        """,
        "viva": [
            ("**What does Histogram Equalization achieve mathematically?**", "It linearizes the Cumulative Distribution Function (CDF), spreading the histogram uniformly."),
            ("**When is Otsu thresholding preferred over manual thresholding?**", "When the image is bimodal and finding the optimal threshold value automatically is required.")
        ]
    },
    "Practical 5: Spatial Filtering & Smoothing": {
        "aim": "Write Python programs using OpenCV to apply spatial domain filters: Averaging, Gaussian, Median, and Bilateral filters.",
        "objectives": [
            "Understand neighborhood convolution using 2D masks.",
            "Reduce noise using low-pass linear smoothing filters.",
            "Compare non-linear median filtering with edge-preserving bilateral filtering."
        ],
        "theory": """
**Spatial Filters:**
- **Averaging Filter ($cv2.blur$):** Uniform box kernel replacing the center pixel with the mean of its neighbors.
- **Gaussian Filter:** Uses a 2D Gaussian bell-curve kernel; weights pixels closer to the center more heavily.
- **Median Filter:** Non-linear filter that replaces the center pixel with the neighborhood median; ideal for salt-and-pepper noise.
- **Bilateral Filter:** Non-linear, edge-preserving filter combining spatial geometric distance with radiometric intensity difference.
        """,
        "viva": [
            ("**Why is Bilateral filtering considered superior for portraits?**", "It smooths flat regions while preserving sharp edge boundaries."),
            ("**Why does Median filter outperform Gaussian for impulse noise?**", "The median is robust against extreme outlier values, whereas Gaussian averages them into the neighborhood.")
        ]
    },
    "Practical 6: Image Restoration & Inpainting": {
        "aim": "Implement image inpainting using the Fast Marching (Telea) and Navier-Stokes (NS) methods to restore corrupted areas.",
        "objectives": [
            "Study image restoration principles and degradation models.",
            "Create binary masks indicating damaged pixel coordinates.",
            "Compare Telea and Navier-Stokes inpainting outputs."
        ],
        "theory": """
**Image Inpainting:**
Restores missing, damaged, or scratched regions using surrounding structural information:
- **Telea Method (Fast Marching):** Propagates known boundary pixels inward, computing a weighted sum based on neighborhood proximity.
- **Navier-Stokes (NS Method):** Uses partial differential equations derived from fluid dynamics to propagate continuous lines of equal intensity (isophotes).
        """,
        "viva": [
            ("**What is the purpose of an inpainting mask?**", "A binary mask where white pixels (255) mark areas to reconstruct, and black pixels (0) represent valid data."),
            ("**Which inpainting method is better for thin scratches?**", "The Telea method is faster and well suited for thin scratches and text removal.")
        ]
    },
    "Practical 7: Lossless vs Lossy Compression": {
        "aim": "Implement image coding techniques to achieve compression and compare original and compressed file sizes.",
        "objectives": [
            "Understand spatial, color, and psychovisual redundancies.",
            "Implement lossy JPEG compression across quality factors.",
            "Evaluate lossless PNG compression and calculate compression ratios."
        ],
        "theory": """
**Compression Theory:**
- **Compression Ratio:** $\\text{Ratio} = \\frac{\\text{Original Size}}{\\text{Compressed Size}}$.
- **Lossy Compression (JPEG):** Uses Discrete Cosine Transform (DCT) and quantization, discarding subtle color variations less visible to the human eye.
- **Lossless Compression (PNG):** Uses predictive filtering and Deflate (LZ77 + Huffman) to achieve exact bit-for-bit reconstruction.
        """,
        "viva": [
            ("**What is the difference between lossy and lossless compression?**", "Lossless allows exact reconstruction; lossy discards psychovisually redundant details for smaller file size."),
            ("**What redundancies exist in digital images?**", "Coding redundancy, inter-pixel (spatial) redundancy, and psychovisual redundancy.")
        ]
    },
    "Practical 8: Morphological Operations": {
        "aim": "Perform morphological operations: Erosion, Dilation, Opening, and Closing on binary images to analyze object boundaries and noise removal.",
        "objectives": [
            "Implement erosion and dilation using structuring elements (kernels).",
            "Apply opening and closing for smoothing and hole-filling.",
            "Extract object contours and compute morphological component areas."
        ],
        "theory": """
**Mathematical Morphology:**
- **Erosion ($A \\ominus B$):** Shrinks object boundaries by keeping pixels where the structuring element completely fits.
- **Dilation ($A \\oplus B$):** Expands object boundaries by setting pixels where the structuring element hits the object.
- **Opening ($A \\circ B$):** Erosion followed by Dilation. Eliminates small isolated noise and thin protrusions.
- **Closing ($A \\bullet B$):** Dilation followed by Erosion. Fills small internal holes and bridges narrow gaps.
        """,
        "viva": [
            ("**What happens when you apply opening to a noisy binary image?**", "Foreground noise specks smaller than the kernel are removed without altering the overall shape of larger objects."),
            ("**What is a structuring element?**", "A small binary matrix that defines the neighborhood shape (rectangle, cross, ellipse) used to probe the input image.")
        ]
    },
    "Practical 9: Correlation-based Object Detection": {
        "aim": "Develop an object detection program using the Correlation Principle (Template Matching).",
        "objectives": [
            "Locate specific objects by sliding a template patch across a target image.",
            "Compute Normalized Cross-Correlation (NCC) metrics.",
            "Threshold correlation peaks and visualize detections with bounding boxes."
        ],
        "theory": """
**Correlation Principle:**
Template matching slides a smaller template $T$ across a larger image $I$:
$$R(x, y) = \\frac{\\sum_{x', y'} (T(x', y') \\cdot I(x+x', y+y'))}{\\sqrt{\\sum T^2 \\cdot \\sum I^2}}$$
OpenCV's $cv2.matchTemplate$ with `cv2.TM_CCOEFF_NORMED` scores matches from $-1.0$ to $+1.0$. Peaks above a user-selected threshold denote detected object locations.
        """,
        "viva": [
            ("**What are the limitations of template matching?**", "It is sensitive to variations in scale, rotation, and lighting changes unless multi-scale templates are evaluated."),
            ("**How is the bounding box drawn around the detected target?**", "Using the coordinate index of the peak match along with the template's width and height.")
        ]
    },
    "Post-Lab 1: Frequency Domain Filtering": {
        "aim": "Compute the 2D Discrete Fourier Transform (DFT) magnitude spectrum and implement ideal low-pass and high-pass frequency domain filters.",
        "objectives": [
            "Transform spatial images into the frequency domain using 2D-DFT.",
            "Inspect low frequencies (centered) and high frequencies (outer edges).",
            "Apply ideal circular frequency masks and reconstruct images using Inverse DFT."
        ],
        "theory": """
**Frequency Domain Processing:**
Decomposes an image into sine and cosine frequency components:
$$F(u, v) = \\sum_{x=0}^{M-1} \\sum_{y=0}^{N-1} f(x, y) e^{-j 2\\pi (\\frac{ux}{M} + \\frac{vy}{N})}$$
- **Low Frequencies:** Smooth surfaces and gradual intensity changes.
- **High Frequencies:** Edges, contours, and fine textures.
- **Ideal Low-Pass Filter (ILPF):** Passes frequencies within radius $D_0$, blurring edges.
- **Ideal High-Pass Filter (IHPF):** Attenuates low frequencies, leaving boundary edges.
        """,
        "viva": [
            ("**Why do we apply fftshift to the DFT output?**", "To move the zero-frequency DC component to the center of the spectrum for easier interpretation and filtering."),
            ("**What visual artifact is caused by ideal frequency filters?**", "Ringing artifacts (Gibbs phenomenon) caused by the sharp cutoff in the frequency domain.")
        ]
    },
    "Post-Lab 2: Advanced Color Spaces & Channels": {
        "aim": "Convert images between RGB, HSV, YCrCb, and CIELAB color spaces and evaluate individual channel components.",
        "objectives": [
            "Decompose images across perceptual and transmission color spaces.",
            "Analyze luminance vs chrominance decoupling.",
            "Inspect individual color channels (Hue, Saturation, Cr, Cb, L*, a*, b*)."
        ],
        "theory": """
**Color Models:**
- **RGB:** Additive hardware display model; couples color and luminance in every channel.
- **HSV:** Perceptual representation dividing color into Hue (type), Saturation (vibrancy), and Value (brightness).
- **YCrCb:** Used in JPEG/MPEG video transmission; decouples Luminance ($Y$) from Chroma differences ($Cr, Cb$).
- **CIELAB:** Perceptually uniform space modeling human vision via Lightness ($L^*$), green-magenta ($a^*$), and blue-yellow ($b^*$).
        """,
        "viva": [
            ("**Why is HSV preferred over RGB for color-based object segmentation?**", "Hue isolates the pure color independently of lighting conditions, making thresholding robust against shadows."),
            ("**Why does YCrCb enable higher compression ratios in JPEG?**", "Human vision is less sensitive to high-frequency color detail than brightness, enabling chroma subsampling.")
        ]
    }
}

# --- Sidebar ---
with st.sidebar:
    st.header("🎛️ Navigation")
    selected_exp = st.selectbox("Select Experiment", list(LAB_METADATA.keys()))
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
    st.caption("Upload any standard JPG or PNG photo to run the experiment.")

# --- Helper Functions ---
def get_download_bytes(img_array):
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

meta = LAB_METADATA[selected_exp]

# --- Main Layout: 3 Tabs ---
tab_app, tab_info, tab_viva = st.tabs(["🚀 Interactive Processing", "🎯 Aim, Objective & Theory", "📝 Viva & Discussion"])

# -------------------------------------------------------------
# TAB 2: AIM, OBJECTIVE & THEORY
# -------------------------------------------------------------
with tab_info:
    st.subheader(f"📌 {selected_exp}")
    st.markdown(f"**Aim:** {meta['aim']}")
    st.markdown("---")
    st.markdown("#### 🎯 Expected Learning Outcomes & Objectives")
    for obj in meta["objectives"]:
        st.markdown(f"- {obj}")
    st.markdown("---")
    st.markdown("#### 📖 Theoretical Background")
    st.markdown(meta["theory"])

# -------------------------------------------------------------
# TAB 3: VIVA & DISCUSSION
# -------------------------------------------------------------
with tab_viva:
    st.subheader("💡 Lab Discussion & Viva Questions")
    for q, a in meta["viva"]:
        with st.expander(q):
            st.write(a)

# -------------------------------------------------------------
# TAB 1: INTERACTIVE WORKBENCH
# -------------------------------------------------------------
with tab_app:
    if uploaded_file is None:
        st.info("👈 Please upload an image from the sidebar to test this experiment interactively.")
    else:
        src_image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(src_image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        rows, cols = img_gray.shape

        st.caption(f"Loaded Image: {cols}x{rows} px | Channels: 3 | Mode: RGB")

        # PRACTICAL 1
        if "Practical 1" in selected_exp:
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Original RGB Image", use_container_width=True)
            with c2:
                mode = st.selectbox("Inspect Channel", ["Grayscale", "Red Channel", "Green Channel", "Blue Channel"])
                if mode == "Grayscale":
                    res = img_gray
                elif mode == "Red Channel":
                    res = img_rgb[:, :, 0]
                elif mode == "Green Channel":
                    res = img_rgb[:, :, 1]
                else:
                    res = img_rgb[:, :, 2]
                st.image(res, caption=f"View: {mode}", use_container_width=True)
                st.download_button("📥 Download Result", get_download_bytes(res), "p1_output.png")

        # PRACTICAL 2
        elif "Practical 2" in selected_exp:
            op_group = st.radio("Select Operation Group", ["Arithmetic Operations", "Bitwise Operations"], horizontal=True)
            if op_group == "Arithmetic Operations":
                st.write("Upload a second image to perform blending or subtraction:")
                f2 = st.file_uploader("Upload Second Image", type=["jpg", "png"], key="p2_f2")
                if f2:
                    img2 = np.array(Image.open(f2).convert("RGB"))
                    img2_res = cv2.resize(img2, (cols, rows))
                    alpha = st.slider("Blending Factor (Alpha)", 0.0, 1.0, 0.5)
                    blended = cv2.addWeighted(img_rgb, alpha, img2_res, 1.0 - alpha, 0)
                    subtracted = cv2.subtract(img_rgb, img2_res)
                    c1, c2 = st.columns(2)
                    c1.image(blended, caption=f"Weighted Addition (Alpha: {alpha:.2f})", use_container_width=True)
                    c2.image(subtracted, caption="cv2.subtract Result", use_container_width=True)
                else:
                    st.warning("Please upload a second image above to test arithmetic operations.")
            else:
                gate = st.selectbox("Bitwise Gate", ["Bitwise NOT", "Bitwise AND (Circular Aperture Mask)", "Bitwise OR"])
                mask = np.zeros((rows, cols), dtype=np.uint8)
                cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 3, 255, -1)
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Source", use_container_width=True)
                if gate == "Bitwise NOT":
                    res = cv2.bitwise_not(img_rgb)
                elif gate == "Bitwise AND (Circular Aperture Mask)":
                    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                else:
                    res = cv2.bitwise_or(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                c2.image(res, caption=gate, use_container_width=True)

        # PRACTICAL 3
        elif "Practical 3" in selected_exp:
            trans = st.selectbox("Transformation Type", ["Translation", "Rotation", "Scaling", "Reflection (Flip)", "Shearing"])
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Original Canvas", use_container_width=True)
            with c2:
                if trans == "Translation":
                    tx = st.slider("Horizontal Shift (X)", -150, 150, 60)
                    ty = st.slider("Vertical Shift (Y)", -150, 150, 40)
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    res = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif trans == "Rotation":
                    deg = st.slider("Rotation Angle (Degrees)", -180, 180, 35)
                    scale = st.slider("Zoom Scaling", 0.4, 1.5, 0.85)
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, scale)
                    res = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif trans == "Scaling":
                    fx = st.slider("Scale X", 0.3, 2.0, 1.2)
                    fy = st.slider("Scale Y", 0.3, 2.0, 0.8)
                    res = cv2.resize(img_rgb, None, fx=fx, fy=fy)
                elif trans == "Reflection (Flip)":
                    axis = st.radio("Direction", ["Horizontal (Left-Right)", "Vertical (Up-Down)"])
                    res = cv2.flip(img_rgb, 1 if "Horizontal" in axis else 0)
                else:
                    sh = st.slider("Shear Factor", 0.1, 0.8, 0.3)
                    M = np.float32([[1, sh, 0], [0, 1, 0], [0, 0, 1]])
                    res = cv2.warpPerspective(img_rgb, M, (int(cols * 1.4), rows))
                st.image(res, caption=f"Transformed: {trans}", use_container_width=True)

        # PRACTICAL 4
        elif "Practical 4" in selected_exp:
            enh = st.selectbox("Enhancement Method", ["Histogram Equalization", "Brightness & Contrast Scaling", "Thresholding"])
            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Original Grayscale", use_container_width=True)
            with c2:
                if enh == "Histogram Equalization":
                    res = cv2.equalizeHist(img_gray)
                    st.image(res, caption="Equalized Image", use_container_width=True)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))
                    ax1.hist(img_gray.ravel(), 256, [0, 256], color="blue")
                    ax1.set_title("Original")
                    ax2.hist(res.ravel(), 256, [0, 256], color="green")
                    ax2.set_title("Equalized")
                    st.pyplot(fig)
                elif enh == "Brightness & Contrast Scaling":
                    alpha = st.slider("Contrast Factor (Alpha)", 0.5, 3.0, 1.5)
                    beta = st.slider("Brightness Offset (Beta)", -100, 100, 30)
                    res = cv2.convertScaleAbs(img_gray, alpha=alpha, beta=beta)
                    st.image(res, caption=f"Contrast={alpha}, Brightness={beta}", use_container_width=True)
                else:
                    th = st.slider("Threshold Value", 0, 255, 128)
                    t_type = st.selectbox("Threshold Technique", ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO"])
                    _, res = cv2.threshold(img_gray, th, 255, getattr(cv2, t_type))
                    st.image(res, caption=f"Result ({t_type})", use_container_width=True)

        # PRACTICAL 5
        elif "Practical 5" in selected_exp:
            f_type = st.selectbox("Select Filter", ["Averaging Filter", "Gaussian Filter", "Median Filter", "Bilateral Filter"])
            k = st.slider("Neighborhood Kernel Size (Odd)", 3, 25, 5, step=2)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Source Image", use_container_width=True)
            with c2:
                if f_type == "Averaging Filter":
                    res = cv2.blur(img_rgb, (k, k))
                elif f_type == "Gaussian Filter":
                    res = cv2.GaussianBlur(img_rgb, (k, k), 0)
                elif f_type == "Median Filter":
                    res = cv2.medianBlur(img_rgb, k)
                else:
                    res = cv2.bilateralFilter(img_rgb, k, 75, 75)
                st.image(res, caption=f"Output ({f_type})", use_container_width=True)

        # PRACTICAL 6
        elif "Practical 6" in selected_exp:
            st.write("Artificial scratch damage simulation and reconstruction:")
            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.line(mask, (cols // 4, rows // 3), (3 * cols // 4, rows // 3), 255, 4)
            cv2.line(mask, (cols // 2, rows // 6), (cols // 2, 5 * rows // 6), 255, 4)
            damaged = img_rgb.copy()
            damaged[mask == 255] = [255, 255, 255]
            alg = st.radio("Inpainting Method", ["Telea Fast Marching (cv2.INPAINT_TELEA)", "Navier-Stokes (cv2.INPAINT_NS)"])
            flag = cv2.INPAINT_TELEA if "Telea" in alg else cv2.INPAINT_NS
            restored = cv2.inpaint(damaged, mask, 3, flag)
            c1, c2, c3 = st.columns(3)
            c1.image(damaged, caption="Simulated Damaged Image", use_container_width=True)
            c2.image(mask, caption="Damage Mask", use_container_width=True)
            c3.image(restored, caption="Restored Output", use_container_width=True)

        # PRACTICAL 7
        elif "Practical 7" in selected_exp:
            q = st.slider("JPEG Quality Factor", 5, 100, 30)
            png_lvl = st.slider("PNG Deflate Compression Level", 0, 9, 9)
            _, enc_jpg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
            _, enc_png = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_lvl])
            raw_kb = (rows * cols * 3) / 1024
            jpg_kb = len(enc_jpg) / 1024
            png_kb = len(enc_png) / 1024
            st.table({
                "Algorithm": ["Raw Uncompressed", f"Lossy JPEG (Quality={q})", f"Lossless PNG (Level={png_lvl})"],
                "File Size": [f"{raw_kb:.1f} KB", f"{jpg_kb:.1f} KB", f"{png_kb:.1f} KB"],
                "Compression Ratio": ["1.00:1", f"{(raw_kb/jpg_kb):.2f}:1", f"{(raw_kb/png_kb):.2f}:1"]
            })
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(cv2.imdecode(enc_jpg, 1), cv2.COLOR_BGR2RGB), caption="JPEG View", use_container_width=True)
            c2.image(cv2.cvtColor(cv2.imdecode(enc_png, 1), cv2.COLOR_BGR2RGB), caption="PNG View", use_container_width=True)

        # PRACTICAL 8
        elif "Practical 8" in selected_exp:
            op = st.selectbox("Morphological Operator", ["Erosion", "Dilation", "Opening", "Closing"])
            k_sz = st.slider("Structuring Element Size", 3, 15, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_sz, k_sz))
            _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            ops = {
                "Erosion": cv2.erode(binary, kernel),
                "Dilation": cv2.dilate(binary, kernel),
                "Opening": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
                "Closing": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            }
            res = ops[op]
            c1, c2 = st.columns(2)
            c1.image(binary, caption="Binary Base", use_container_width=True)
            c2.image(res, caption=f"Result: {op}", use_container_width=True)

        # PRACTICAL 9
        elif "Practical 9" in selected_exp:
            st.write("Extract a center region from the image to use as template:")
            tw = st.slider("Template Width", 40, 160, 80)
            th = st.slider("Template Height", 40, 160, 80)
            template = img_gray[rows // 2:rows // 2 + th, cols // 2:cols // 2 + tw]
            corr = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = st.slider("Match Threshold", 0.5, 0.99, 0.8)
            loc = np.where(corr >= threshold)
            detected = img_rgb.copy()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(detected, pt, (pt[0] + tw, pt[1] + th), (255, 230, 0), 2)
            c1, c2 = st.columns([1, 3])
            c1.image(template, caption="Template Sub-Window", use_container_width=True)
            c2.image(detected, caption="Detected Template Locations", use_container_width=True)

        # POST-LAB 1
        elif "Post-Lab 1" in selected_exp:
            dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            mag = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            f_type = st.radio("Frequency Filter", ["Spectrum Only", "Ideal Low-Pass Filter", "Ideal High-Pass Filter"], horizontal=True)
            if f_type == "Spectrum Only":
                c1, c2 = st.columns(2)
                c1.image(img_gray, caption="Grayscale Spatial Image", use_container_width=True)
                c2.image(norm_mag, caption="2D-DFT Frequency Spectrum", use_container_width=True)
            else:
                radius = st.slider("Cutoff Radius (D0)", 10, min(rows, cols) // 2, 40)
                crow, ccol = rows // 2, cols // 2
                y, x = np.ogrid[:rows, :cols]
                dist = np.sqrt((x - ccol)**2 + (y - crow)**2)
                mask = dist <= radius if "Low-Pass" in f_type else dist > radius
                fshift = dft_shift.copy()
                fshift[:, :, 0] *= mask
                fshift[:, :, 1] *= mask
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
                cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
                c1, c2, c3 = st.columns(3)
                c1.image(img_gray, caption="Original Gray", use_container_width=True)
                c2.image(mask.astype(np.uint8) * 255, caption="Filter Mask", use_container_width=True)
                c3.image(img_back.astype(np.uint8), caption="Reconstructed (Inverse DFT)", use_container_width=True)

        # POST-LAB 2
        else:
            space = st.selectbox("Target Color Space", ["HSV", "YCrCb", "CIELAB"])
            if space == "HSV":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                labels = ["Hue (H)", "Saturation (S)", "Value (V)"]
            elif space == "YCrCb":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                labels = ["Luminance (Y)", "Chroma Red (Cr)", "Chroma Blue (Cb)"]
            else:
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                labels = ["Lightness (L)", "Green-Red (a*)", "Blue-Yellow (b*)"]
            c1, c2, c3 = cv2.split(converted)
            st.image(converted, caption=f"Full {space} Color Space Representation", use_container_width=True)
            ca, cb, cc = st.columns(3)
            ca.image(c1, caption=labels[0], use_container_width=True)
            cb.image(c2, caption=labels[1], use_container_width=True)
            cc.image(c3, caption=labels[2], use_container_width=True)
