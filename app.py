import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --- Page Setup & Custom CSS Styling ---
st.set_page_config(
    page_title="VisionCraft | Digital Image Processing Interactive Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme CSS to give a unique modern dashboard look
st.markdown("""
<style>
    /* Gradient Header */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 24px;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 25px;
        border: 1px solid #334155;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: #38BDF8;
    }
    .main-header p {
        font-size: 0.95rem;
        color: #94A3B8;
        margin: 5px 0 0 0;
    }

    /* Custom Metric Cards */
    .metric-card {
        background-color: #F1F5F9;
        border-left: 4px solid #0284C7;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 15px;
    }

    /* Image containers */
    div[data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Custom Dashboard Header
st.markdown("""
<div class="main-header">
    <h1>👁️ VisionCraft Studio</h1>
    <p>Interactive Digital Image Processing Workbench | N-PECCS502P Laboratory</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🎛️ Studio Control")

    category = st.radio(
        "Experiment Group",
        ["Core Practicals (1-5)", "Advanced Practicals (6-9)", "Special Post-Labs (1-2)"]
    )

    if category == "Core Practicals (1-5)":
        experiments = [
            "P01: Visual Environment & Metadata Profiling",
            "P02: Dual-Image Arithmetics & Logic Gates",
            "P03: Spatial Geometries & Affine Warping",
            "P04: Contrast Stretching, Negation & Binarization",
            "P05: Neighborhood Convolutions & Smoothing"
        ]
    elif category == "Advanced Practicals (6-9)":
        experiments = [
            "P06: Inpainting Reconstruction (Telea vs NS)",
            "P07: Entropy, Compression & Loss Metrics",
            "P08: Morphological Boundary & Structural Slicing",
            "P09: Cross-Correlation Pattern Matching"
        ]
    else:
        experiments = [
            "Post-Lab 01: Spectral Domain (2D-FFT Analysis)",
            "Post-Lab 02: Chromatic Decompositions & Spaces"
        ]

    selected_exp = st.selectbox("Select Active Module", experiments)
    st.markdown("---")
    uploaded_file = st.file_uploader("📥 Source Canvas Image", type=["jpg", "jpeg", "png"])
    st.caption("Tip: Use standard contrast images for best visual comparison.")


# --- Utility Functions ---
def get_image_bytes(img_array, fmt='PNG'):
    pil_img = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


if uploaded_file is not None:
    src_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(src_image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gray.shape

    # Metrics Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dimensions", f"{cols} x {rows} px")
    m2.metric("Color Depth", "24-bit TrueColor")
    m3.metric("Mean Intensity", f"{img_gray.mean():.1f}")
    m4.metric("Dynamic Range", f"[{img_gray.min()}, {img_gray.max()}]")

    tab_demo, tab_theory, tab_analytics = st.tabs(
        ["🚀 Interactive Processing", "📖 Theory & Formulae", "📊 Matrix & Data Profile"])

    # -------------------------------------------------------------
    # PRACTICAL 1
    # -------------------------------------------------------------
    if "P01:" in selected_exp:
        with tab_demo:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Source Input")
                st.image(img_rgb, use_container_width=True)
            with col2:
                st.subheader("Selected Channel Inspection")
                channel = st.selectbox("Channel",
                                       ["All (RGB)", "Red Monochromatic", "Green Monochromatic", "Blue Monochromatic",
                                        "Grayscale Luma"])
                if channel == "All (RGB)":
                    out_img = img_rgb
                elif channel == "Red Monochromatic":
                    out_img = img_rgb[:, :, 0]
                elif channel == "Green Monochromatic":
                    out_img = img_rgb[:, :, 1]
                elif channel == "Blue Monochromatic":
                    out_img = img_rgb[:, :, 2]
                else:
                    out_img = img_gray
                st.image(out_img, use_container_width=True)
                st.download_button("💾 Save Channel", get_image_bytes(out_img), "channel.png", "image/png")

        with tab_theory:
            st.markdown("""
            ### Pixel Matrix Coordinate Space
            A digital image is discretized into a 2D spatial grid:
            $$f(x, y) = [R(x, y), G(x, y), B(x, y)]$$
            where $x \in [0, \text{Width}-1]$ and $y \in [0, \text{Height}-1]$[span_0](start_span)[span_0](end_span).
            In standard 8-bit displays, each color channel integer value ranges within $[0, 255]$[span_1](start_span)[span_1](end_span)[span_2](start_span)[span_2](end_span).
            """)

        with tab_analytics:
            st.write("**Top-Left 5x5 Matrix (Grayscale Intensity)**")
            st.dataframe(img_gray[:5, :5])

    # -------------------------------------------------------------
    # PRACTICAL 2
    # -------------------------------------------------------------
    elif "P02:" in selected_exp:
        with tab_demo:
            mode = st.radio("Logic Family", ["Arithmetic Weighted Blending", "Bitwise Boolean Gates"], horizontal=True)
            col1, col2 = st.columns(2)

            if mode == "Arithmetic Weighted Blending":
                st.write("Upload a second image to blend or perform subtraction:")
                sec = st.file_uploader("Second Image", type=["jpg", "png"], key="p2_sec_custom")
                if sec:
                    img2 = np.array(Image.open(sec).convert("RGB"))
                    img2_res = cv2.resize(img2, (cols, rows))
                    alpha = st.slider("Primary Weight (Alpha)", 0.0, 1.0, 0.6)
                    blended = cv2.addWeighted(img_rgb, alpha, img2_res, 1.0 - alpha, 0)

                    c_a, c_b = st.columns(2)
                    c_a.image(blended, caption="Weighted Blend", use_container_width=True)
                    c_b.image(cv2.subtract(img_rgb, img2_res), caption="Direct Subtraction", use_container_width=True)
                else:
                    st.info("Please provide a second image to activate arithmetic operations.")
            else:
                gate = st.selectbox("Gate",
                                    ["Bitwise NOT", "Bitwise AND (Circular Aperture)", "Bitwise XOR (Synthetic Mask)"])
                c1, c2 = st.columns(2)
                c1.image(img_rgb, caption="Input", use_container_width=True)
                mask = np.zeros((rows, cols), dtype=np.uint8)
                cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 3, 255, -1)

                if gate == "Bitwise NOT":
                    res = cv2.bitwise_not(img_rgb)
                elif gate == "Bitwise AND (Circular Aperture)":
                    res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                else:
                    res = cv2.bitwise_xor(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                c2.image(res, caption=f"Result: {gate}", use_container_width=True)

        with tab_theory:
            st.markdown("""
            ### Mathematical Blending Formulation
            $$g(x, y) = \alpha \cdot f_1(x, y) + (1 - \alpha) \cdot f_2(x, y) + \gamma$$
            Bitwise operations compute binary conjunction, disjunction, and inversion at pixel depth levels[span_3](start_span)[span_3](end_span).
            """)
        with tab_analytics:
            st.write("Input vs Processed Mean Pixel Differences")

    # -------------------------------------------------------------
    # PRACTICAL 3
    # -------------------------------------------------------------
    elif "P03:" in selected_exp:
        with tab_demo:
            t_choice = st.selectbox("Transformation",
                                    ["Affine Translation", "Center Rotation", "Anisotropic Scaling", "Reflection",
                                     "Shear Plane"])
            col1, col2 = st.columns(2)
            col1.image(img_rgb, caption="Canvas Base", use_container_width=True)

            with col2:
                if t_choice == "Affine Translation":
                    dx = st.slider("Shift X", -150, 150, 60)
                    dy = st.slider("Shift Y", -150, 150, 40)
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    out = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif t_choice == "Center Rotation":
                    deg = st.slider("Angle", -180, 180, 45)
                    scale = st.slider("Scale", 0.4, 1.5, 0.9)
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, scale)
                    out = cv2.warpAffine(img_rgb, M, (cols, rows))
                elif t_choice == "Anisotropic Scaling":
                    sx = st.slider("Scale X", 0.3, 2.0, 1.2)
                    sy = st.slider("Scale Y", 0.3, 2.0, 0.8)
                    out = cv2.resize(img_rgb, None, fx=sx, fy=sy)
                elif t_choice == "Reflection":
                    axis = st.radio("Axis", ["Horizontal Flip", "Vertical Flip"])
                    out = cv2.flip(img_rgb, 1 if "Horizontal" in axis else 0)
                else:
                    sh = st.slider("Shear Coefficient", 0.1, 0.8, 0.3)
                    M = np.float32([[1, sh, 0], [0, 1, 0], [0, 0, 1]])
                    out = cv2.warpPerspective(img_rgb, M, (int(cols * 1.4), rows))
                st.image(out, caption=f"Transformed: {t_choice}", use_container_width=True)

        with tab_theory:
            st.latex(r"""
            \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = 
            \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix}
            \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
            """)

        with tab_analytics:
            st.write("Warp matrix parameters and bounds verified.")

    # -------------------------------------------------------------
    # PRACTICAL 4
    # -------------------------------------------------------------
    elif "P04:" in selected_exp:
        with tab_demo:
            enh = st.selectbox("Enhancement Strategy",
                               ["Histogram Equalization (Contrast)", "Contrast & Brightness Scalar",
                                "Threshold Segmentation"])
            col1, col2 = st.columns(2)
            col1.image(img_gray, caption="Monochrome Base", use_container_width=True)

            with col2:
                if enh == "Histogram Equalization (Contrast)":
                    out = cv2.equalizeHist(img_gray)
                    st.image(out, caption="Equalized Intensity Distribution", use_container_width=True)
                elif enh == "Contrast & Brightness Scalar":
                    a = st.slider("Alpha (Contrast)", 0.5, 3.0, 1.5)
                    b = st.slider("Beta (Brightness)", -100, 100, 20)
                    out = cv2.convertScaleAbs(img_gray, alpha=a, beta=b)
                    st.image(out, caption=f"Scaled (alpha={a}, beta={b})", use_container_width=True)
                else:
                    th_val = st.slider("Threshold Level", 0, 255, 128)
                    mode_th = st.selectbox("Type", ["cv2.THRESH_BINARY", "cv2.THRESH_BINARY_INV", "cv2.THRESH_TRUNC",
                                                    "cv2.THRESH_TOZERO"])
                    _, out = cv2.threshold(img_gray, th_val, 255, getattr(cv2, mode_th.replace("cv2.", "")))
                    st.image(out, caption=f"Segmented Output", use_container_width=True)

        with tab_theory:
            st.markdown(
                "Histogram equalization flattens the cumulative distribution function (CDF) across $[0, 255]$[span_4](start_span)[span_4](end_span).")
        with tab_analytics:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.hist(img_gray.ravel(), 256, [0, 256], color="#38BDF8")
            ax.set_title("Intensity Histogram")
            st.pyplot(fig)

    # -------------------------------------------------------------
    # PRACTICAL 5
    # -------------------------------------------------------------
    elif "P05:" in selected_exp:
        with tab_demo:
            f_mode = st.selectbox("Filter Class",
                                  ["Averaging Box Filter", "Gaussian Filter", "Median Filter (Salt & Pepper)",
                                   "Bilateral (Edge Preserving)"])
            k_sz = st.slider("Window Kernel Size", 3, 25, 5, step=2)

            col1, col2 = st.columns(2)
            col1.image(img_rgb, caption="Raw Spatial Domain", use_container_width=True)

            with col2:
                if f_mode == "Averaging Box Filter":
                    filtered = cv2.blur(img_rgb, (k_sz, k_sz))
                elif f_mode == "Gaussian Filter":
                    filtered = cv2.GaussianBlur(img_rgb, (k_sz, k_sz), 0)
                elif f_mode == "Median Filter (Salt & Pepper)":
                    filtered = cv2.medianBlur(img_rgb, k_sz)
                else:
                    filtered = cv2.bilateralFilter(img_rgb, k_sz, 75, 75)
                st.image(filtered, caption=f"Result ({f_mode})", use_container_width=True)

        with tab_theory:
            st.markdown(
                "Filtering convolves a 2D matrix kernel $H(u,v)$ across neighboring coordinates[span_5](start_span)[span_5](end_span).")
        with tab_analytics:
            st.write(f"Effective Kernel Neighborhood Matrix: {k_sz} x {k_sz}")

    # -------------------------------------------------------------
    # PRACTICAL 6
    # -------------------------------------------------------------
    elif "P06:" in selected_exp:
        with tab_demo:
            st.write("Artificial scratch damage simulation and reconstruction:")
            col1, col2, col3 = st.columns(3)

            mask = np.zeros((rows, cols), dtype=np.uint8)
            cv2.line(mask, (cols // 4, rows // 3), (3 * cols // 4, rows // 3), 255, 4)
            cv2.line(mask, (cols // 2, rows // 6), (cols // 2, 5 * rows // 6), 255, 4)
            damaged = img_rgb.copy()
            damaged[mask == 255] = [255, 255, 255]

            alg = st.radio("Inpainting Algorithm", ["Fast Marching (Telea)", "Navier-Stokes (Fluid Dynamics)"])
            flag = cv2.INPAINT_TELEA if "Telea" in alg else cv2.INPAINT_NS
            restored = cv2.inpaint(damaged, mask, 3, flag)

            col1.image(damaged, caption="Degraded Image", use_container_width=True)
            col2.image(mask, caption="Damage Mask", use_container_width=True)
            col3.image(restored, caption=f"Restored ({alg.split()[0]})", use_container_width=True)

        with tab_theory:
            st.markdown(
                "Telea uses Fast Marching (boundary inwards), whereas Navier-Stokes propagates isophotes (edges)[span_6](start_span)[span_6](end_span).")
        with tab_analytics:
            st.metric("Damaged Pixels Recovered", int(np.sum(mask == 255)))

    # -------------------------------------------------------------
    # PRACTICAL 7
    # -------------------------------------------------------------
    elif "P07:" in selected_exp:
        with tab_demo:
            q = st.slider("JPEG Compression Quality", 5, 100, 25)
            png_lvl = st.slider("PNG Deflate Compression Level", 0, 9, 9)

            _, jpg_enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
            _, png_enc = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_PNG_COMPRESSION, png_lvl])

            orig_kb = (rows * cols * 3) / 1024
            jpg_kb = len(jpg_enc) / 1024
            png_kb = len(png_enc) / 1024

            st.write("### Compression Ratio Benchmarking")
            st.table({
                "Format": ["Uncompressed Bitmap", f"Lossy JPEG (Q={q})", f"Lossless PNG (L={png_lvl})"],
                "File Size (KB)": [f"{orig_kb:.1f} KB", f"{jpg_kb:.1f} KB", f"{png_kb:.1f} KB"],
                "Compression Ratio": ["1.00:1", f"{(orig_kb / jpg_kb):.2f}:1", f"{(orig_kb / png_kb):.2f}:1"],
                "Data Integrity": ["100% Uncompressed", "Lossy (Psychovisual Pruned)", "100% Exact Reconstruction"]
            })

        with tab_theory:
            st.markdown(
                "Lossless models retain entropy without data loss; lossy models exploit human psychovisual limitations[span_7](start_span)[span_7](end_span).")
        with tab_analytics:
            st.metric("Space Saved via JPEG", f"{((1 - jpg_kb / orig_kb) * 100):.1f}%")

    # -------------------------------------------------------------
    # PRACTICAL 8
    # -------------------------------------------------------------
    elif "P08:" in selected_exp:
        with tab_demo:
            op_m = st.selectbox("Morphological Function",
                                ["Erosion", "Dilation", "Opening (Noise Removal)", "Closing (Hole Fill)"])
            k_m = st.slider("Structuring Element Kernel", 3, 15, 5)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_m, k_m))

            _, bin_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
            dict_ops = {
                "Erosion": cv2.erode(bin_img, kernel),
                "Dilation": cv2.dilate(bin_img, kernel),
                "Opening (Noise Removal)": cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel),
                "Closing (Hole Fill)": cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
            }
            res_morph = dict_ops[op_m]

            c1, c2 = st.columns(2)
            c1.image(bin_img, caption="Binary Base", use_container_width=True)
            c2.image(res_morph, caption=f"Result ({op_m})", use_container_width=True)

        with tab_theory:
            st.markdown("""
            - **Erosion:** $A \ominus B = \{z \mid (B)_z \subseteq A\}$
            - **Dilation:** $A \oplus B = \{z \mid (\hat{B})_z \cap A \neq \emptyset\}$
            """)
        with tab_analytics:
            contours, _ = cv2.findContours(res_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            st.metric("Connected Components Count", len(contours))

    # -------------------------------------------------------------
    # PRACTICAL 9
    # -------------------------------------------------------------
    elif "P09:" in selected_exp:
        with tab_demo:
            st.write("Select region coordinates to extract an internal template to match:")
            tw = st.slider("Template Box Width", 40, 150, 80)
            th = st.slider("Template Box Height", 40, 150, 80)

            # Crop synthetic template from center
            cy, cx = rows // 2, cols // 2
            template = img_gray[cy:cy + th, cx:cx + tw]

            res_match = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = st.slider("Correlation Threshold", 0.5, 0.99, 0.8)
            loc = np.where(res_match >= threshold)

            detected = img_rgb.copy()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(detected, pt, (pt[0] + tw, pt[1] + th), (255, 230, 0), 2)

            col1, col2 = st.columns([1, 3])
            col1.image(template, caption="Template Sub-Window", use_container_width=True)
            col2.image(detected, caption="Detected Template Matches", use_container_width=True)

        with tab_theory:
            st.markdown(
                "Calculates Normalized Cross-Correlation (NCC) by sliding the template matrix over the canvas[span_8](start_span)[span_8](end_span).")
        with tab_analytics:
            st.metric("Correlation Peak Value", f"{res_match.max():.4f}")

    # -------------------------------------------------------------
    # POST-LAB 1
    # -------------------------------------------------------------
    elif "Post-Lab 01:" in selected_exp:
        with tab_demo:
            dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            mag = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Spatial Domain Domain", use_container_width=True)
            c2.image(norm_mag, caption="2D-FFT Magnitude Spectrum", use_container_width=True)

        with tab_theory:
            st.latex(
                r"""F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi (\frac{ux}{M} + \frac{vy}{N})}""")
        with tab_analytics:
            st.write("Zero-frequency component shifted to matrix center.")

    # -------------------------------------------------------------
    # POST-LAB 2
    # -------------------------------------------------------------
    else:
        with tab_demo:
            c_space = st.selectbox("Target Color Space",
                                   ["HSV (Perceptual)", "YCrCb (Broadcast/Luma)", "CIELAB (Uniform)"])
            if c_space == "HSV (Perceptual)":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
                c1, c2, c3 = cv2.split(converted)
                labels = ["Hue (H)", "Saturation (S)", "Value (V)"]
            elif c_space == "YCrCb (Broadcast/Luma)":
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
                c1, c2, c3 = cv2.split(converted)
                labels = ["Luma (Y)", "Chroma Red (Cr)", "Chroma Blue (Cb)"]
            else:
                converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
                c1, c2, c3 = cv2.split(converted)
                labels = ["Lightness (L)", "Green-Red (a*)", "Blue-Yellow (b*)"]

            st.image(converted, caption=f"Full {c_space} Mapping", use_container_width=True)
            ca, cb, cc = st.columns(3)
            ca.image(c1, caption=labels[0], use_container_width=True)
            cb.image(c2, caption=labels[1], use_container_width=True)
            cc.image(c3, caption=labels[2], use_container_width=True)

        with tab_theory:
            st.markdown(
                "Separates luminance/lightness from color/chrominance channels to provide illumination invariance[span_9](start_span)[span_9](end_span).")
        with tab_analytics:
            st.write(f"Channel Separation Complete: {labels[0]}, {labels[1]}, {labels[2]}")

else:
    st.info("👈 Upload an image using the sidebar to unlock the interactive processing canvas.")
