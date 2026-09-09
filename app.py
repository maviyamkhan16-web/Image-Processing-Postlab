import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="SBJITMR  Lab Portal", layout="wide")

st.title("🔬 Digital Image Processing Lab Portal (N-PECCS502P)")
st.caption("Department of Computer Science and Engineering | S. B. Jain Institute of Technology, Management & Research")

# All 9 Practicals + 2 Post-Labs
experiments = [
    "Practical 1: Introduction, Reading & Displaying Images",
    "Practical 2: Color Formats, Arithmetic & Bitwise Operations",
    "Practical 3: 2-D Geometric Transformations (Affine, Reflection, Shearing)",
    "Practical 4: Spatial Domain Enhancements & Histogram Equalization",
    "Practical 5: Spatial Filtering (Averaging, Gaussian, Median, Bilateral)",
    "Practical 6: Image Inpainting & Restoration (Telea, NS, Denoising)",
    "Practical 7: Lossy vs Lossless Image Compression (JPEG/PNG/RLE/LZW)",
    "Practical 8: Morphological Operations (Erosion, Dilation, Opening, Closing)",
    "Practical 9: Object Detection via Correlation (Template Matching)",
    "Post-Lab 1: Frequency Domain Filtering (2D-DFT Magnitude Spectrum)",
    "Post-Lab 2: Advanced Color Spaces (RGB, HSV, YCrCb, LAB & Channels)"
]

selected_exp = st.sidebar.selectbox("Select Experiment", experiments)
st.sidebar.markdown("---")

# Main Image Uploader
uploaded_file = st.sidebar.file_uploader("Upload Primary Input Image", type=["jpg", "jpeg", "png"])


def load_cv_image(file):
    image = Image.open(file)
    rgb = np.array(image.convert("RGB"))
    return rgb


if uploaded_file is not None:
    img_rgb = load_cv_image(uploaded_file)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gray.shape

    # -------------------------------------------------------------
    # PRACTICAL 1
    # -------------------------------------------------------------
    if "Practical 1:" in selected_exp:
        st.header("Practical 1: Setup & Basic Image I/O")
        st.info("**Aim:** Introduction to Python, OpenCV, NumPy, and basic image loading and dimensions inspection.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Loaded Image")
            st.image(img_rgb, use_container_width=True)
        with c2:
            st.subheader("Image Metadata")
            st.write(f"- **Resolution:** {cols} x {rows} pixels")
            st.write(f"- **Channels:** {img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1}")
            st.write(f"- **Data Type:** `{img_rgb.dtype}`")
            st.write(f"- **Max Intensity:** {img_rgb.max()} | **Min Intensity:** {img_rgb.min()}")

    # -------------------------------------------------------------
    # PRACTICAL 2
    # -------------------------------------------------------------
    elif "Practical 2:" in selected_exp:
        st.header("Practical 2: Arithmetic and Bitwise Operations")
        st.info("**Aim:** Perform image arithmetic (Addition, Subtraction) and Bitwise logic (AND, OR, XOR, NOT).")

        op_type = st.radio("Choose Operation Group", ["Arithmetic Operations", "Bitwise Operations"])

        if op_type == "Arithmetic Operations":
            st.write("Upload a second image of any resolution (it will be automatically resized for arithmetic).")
            sec_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"], key="p2_sec")

            if sec_file:
                img2_rgb = load_cv_image(sec_file)
                img2_resized = cv2.resize(img2_rgb, (cols, rows))

                arith_mode = st.selectbox("Operation",
                                          ["Weighted Addition (addWeighted)", "Direct Subtraction (subtract)"])
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_rgb, caption="Image 1", use_container_width=True)
                with c2:
                    st.image(img2_resized, caption="Image 2 (Resized)", use_container_width=True)
                with c3:
                    if arith_mode == "Weighted Addition (addWeighted)":
                        alpha = st.slider("Weight Image 1", 0.0, 1.0, 0.5)
                        beta = 1.0 - alpha
                        res = cv2.addWeighted(img_rgb, alpha, img2_resized, beta, 0)
                        st.image(res, caption=f"Blend (Alpha: {alpha:.2f}, Beta: {beta:.2f})", use_container_width=True)
                    else:
                        res = cv2.subtract(img_rgb, img2_resized)
                        st.image(res, caption="cv2.subtract Result", use_container_width=True)

        else:  # Bitwise
            bitwise_choice = st.selectbox("Bitwise Logic",
                                          ["Bitwise NOT", "Bitwise AND (with itself/mask)", "Bitwise OR",
                                           "Bitwise XOR"])
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_rgb, caption="Original Image", use_container_width=True)
            with c2:
                if bitwise_choice == "Bitwise NOT":
                    res = cv2.bitwise_not(img_rgb)
                    st.image(res, caption="Bitwise NOT Output", use_container_width=True)
                else:
                    # Synthetic circular mask for demonstration
                    mask = np.zeros((rows, cols), dtype=np.uint8)
                    cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 3, 255, -1)
                    if bitwise_choice == "Bitwise AND (with itself/mask)":
                        res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                    elif bitwise_choice == "Bitwise OR":
                        res = cv2.bitwise_or(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                    else:
                        res = cv2.bitwise_xor(img_rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
                    st.image(res, caption=f"{bitwise_choice} Output", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 3
    # -------------------------------------------------------------
    elif "Practical 3:" in selected_exp:
        st.header("Practical 3: 2-D Geometric Transformations")
        st.info("**Aim:** Apply Translation, Rotation, Scaling, Reflection, Shearing, and Cropping.")

        geom_op = st.selectbox("Select Transformation", [
            "Translation", "Rotation", "Scaling", "Vertical/Horizontal Reflection", "Shearing (X and Y)", "Cropping"
        ])

        c1, c2 = st.columns(2)
        with c1:
            st.image(img_rgb, caption="Original", use_container_width=True)
        with c2:
            if geom_op == "Translation":
                tx = st.slider("X Shift (Pixels)", -200, 200, 100)
                ty = st.slider("Y Shift (Pixels)", -200, 200, 50)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                dst = cv2.warpAffine(img_rgb, M, (cols, rows))
                st.image(dst, caption=f"Shifted by ({tx}, {ty})", use_container_width=True)

            elif geom_op == "Rotation":
                angle = st.slider("Angle (Degrees)", -180, 180, 30)
                scale = st.slider("Scale inside Rotation", 0.2, 2.0, 0.8)
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
                dst = cv2.warpAffine(img_rgb, M, (cols, rows))
                st.image(dst, caption=f"Rotated {angle}°", use_container_width=True)

            elif geom_op == "Scaling":
                fx = st.slider("Width Scale Factor", 0.2, 3.0, 1.5)
                fy = st.slider("Height Scale Factor", 0.2, 3.0, 1.5)
                dst = cv2.resize(img_rgb, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
                st.image(dst, caption=f"Resized: {dst.shape[1]}x{dst.shape[0]}", use_container_width=True)

            elif geom_op == "Vertical/Horizontal Reflection":
                flip_mode = st.radio("Flip Direction", ["Vertical Flip (Up-Down)", "Horizontal Flip (Left-Right)"])
                code = 0 if "Vertical" in flip_mode else 1
                dst = cv2.flip(img_rgb, code)
                st.image(dst, caption=flip_mode, use_container_width=True)

            elif geom_op == "Shearing (X and Y)":
                shear_axis = st.radio("Shear Axis", ["X-Axis", "Y-Axis"])
                factor = st.slider("Shear Factor", 0.1, 1.0, 0.5)
                if shear_axis == "X-Axis":
                    M = np.float32([[1, factor, 0], [0, 1, 0], [0, 0, 1]])
                else:
                    M = np.float32([[1, 0, 0], [factor, 1, 0], [0, 0, 1]])
                dst = cv2.warpPerspective(img_rgb, M, (int(cols * 1.5), int(rows * 1.5)))
                st.image(dst, caption=f"Sheared along {shear_axis}", use_container_width=True)

            elif geom_op == "Cropping":
                st.write("Specify ROI bounding box coordinates:")
                y1 = st.slider("Start Row (Y1)", 0, rows // 2, 50)
                y2 = st.slider("End Row (Y2)", rows // 2, rows, rows - 50)
                x1 = st.slider("Start Col (X1)", 0, cols // 2, 50)
                x2 = st.slider("End Col (X2)", cols // 2, cols, cols - 50)
                cropped = img_rgb[y1:y2, x1:x2]
                st.image(cropped, caption=f"Cropped ({x2 - x1}x{y2 - y1})", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 4
    # -------------------------------------------------------------
    elif "Practical 4:" in selected_exp:
        st.header("Practical 4: Spatial Domain Enhancements")
        st.info(
            "**Aim:** Implement Negative, Brightness/Contrast Adjustment, Histogram Equalization, and Thresholding.")

        enh_mode = st.selectbox("Select Enhancement Technique", [
            "Negative Transformation",
            "Brightness & Contrast (convertScaleAbs)",
            "Histogram Equalization & Intensity Graph",
            "Thresholding (Binary, Truncate, ToZero)"
        ])

        if enh_mode == "Negative Transformation":
            neg = 255 - img_rgb
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Original", use_container_width=True)
            c2.image(neg, caption="Negative Image", use_container_width=True)

        elif enh_mode == "Brightness & Contrast (convertScaleAbs)":
            alpha = st.slider("Contrast (Alpha)", 0.5, 3.0, 1.5)
            beta = st.slider("Brightness (Beta)", -100, 100, 20)
            res = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)
            c1, c2 = st.columns(2)
            c1.image(img_rgb, caption="Original", use_container_width=True)
            c2.image(res, caption=f"Alpha={alpha}, Beta={beta}", use_container_width=True)

        elif "Histogram Equalization" in enh_mode:
            equ = cv2.equalizeHist(img_gray)
            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Original Grayscale", use_container_width=True)
            c2.image(equ, caption="Histogram Equalized", use_container_width=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
            ax1.hist(img_gray.ravel(), 256, [0, 256], color='blue')
            ax1.set_title("Original Histogram")
            ax2.hist(equ.ravel(), 256, [0, 256], color='green')
            ax2.set_title("Equalized Histogram")
            st.pyplot(fig)

        elif "Thresholding" in enh_mode:
            thresh_type = st.selectbox("OpenCV Threshold Type", [
                "THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"
            ])
            val = st.slider("Threshold Value", 0, 255, 128)
            mapping = {
                "THRESH_BINARY": cv2.THRESH_BINARY,
                "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
                "THRESH_TRUNC": cv2.THRESH_TRUNC,
                "THRESH_TOZERO": cv2.THRESH_TOZERO,
                "THRESH_TOZERO_INV": cv2.THRESH_TOZERO_INV
            }
            _, th_out = cv2.threshold(img_gray, val, 255, mapping[thresh_type])
            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Original Gray", use_container_width=True)
            c2.image(th_out, caption=f"Threshold Result ({thresh_type})", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 5
    # -------------------------------------------------------------
    elif "Practical 5:" in selected_exp:
        st.header("Practical 5: Spatial Filtering")
        st.info("**Aim:** Apply Averaging, Gaussian, Median, and Bilateral spatial domain smoothing filters.")

        filt = st.selectbox("Filter Choice",
                            ["Averaging Filter (cv2.blur)", "Gaussian Filter", "Median Filter", "Bilateral Filter"])
        k = st.slider("Kernel / Window Size (Odd)", 3, 25, 5, step=2)

        c1, c2 = st.columns(2)
        c1.image(img_rgb, caption="Original", use_container_width=True)

        with c2:
            if filt == "Averaging Filter (cv2.blur)":
                res = cv2.blur(img_rgb, (k, k))
                st.image(res, caption=f"Averaging Filter ({k}x{k})", use_container_width=True)
            elif filt == "Gaussian Filter":
                sigma = st.slider("Gaussian Sigma", 0.1, 10.0, 1.5)
                res = cv2.GaussianBlur(img_rgb, (k, k), sigma)
                st.image(res, caption=f"Gaussian Filter ({k}x{k}, sigma={sigma})", use_container_width=True)
            elif filt == "Median Filter":
                res = cv2.medianBlur(img_rgb, k)
                st.image(res, caption=f"Median Filter ({k}x{k})", use_container_width=True)
            else:
                d = st.slider("Diameter (d)", 3, 15, 9)
                sc = st.slider("Sigma Color", 10, 150, 75)
                ss = st.slider("Sigma Space", 10, 150, 75)
                res = cv2.bilateralFilter(img_rgb, d, sc, ss)
                st.image(res, caption="Bilateral Filter (Edge-Preserving)", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 6
    # -------------------------------------------------------------
    elif "Practical 6:" in selected_exp:
        st.header("Practical 6: Image Restoration & Inpainting")
        st.info("**Aim:** Remove damaged scratches using Telea and Navier-Stokes inpainting algorithms.")

        st.write("Simulate damage by generating interactive artificial scratch marks on your image:")
        scratch_thickness = st.slider("Scratch Thickness", 2, 10, 3)

        # Artificial damage
        damaged_img = img_rgb.copy()
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.line(damaged_img, (cols // 4, rows // 3), (3 * cols // 4, rows // 3), (0, 0, 0), scratch_thickness)
        cv2.line(damaged_img, (cols // 2, rows // 5), (cols // 2, 4 * rows // 5), (0, 0, 0), scratch_thickness)
        cv2.line(mask, (cols // 4, rows // 3), (3 * cols // 4, rows // 3), 255, scratch_thickness)
        cv2.line(mask, (cols // 2, rows // 5), (cols // 2, 4 * rows // 5), 255, scratch_thickness)

        inpaint_method = st.radio("Inpainting Algorithm",
                                  ["Telea Method (cv2.INPAINT_TELEA)", "Navier-Stokes Method (cv2.INPAINT_NS)"])
        flag = cv2.INPAINT_TELEA if "Telea" in inpaint_method else cv2.INPAINT_NS
        restored = cv2.inpaint(damaged_img, mask, 3, flag)

        c1, c2, c3 = st.columns(3)
        c1.image(damaged_img, caption="Simulated Damaged Image", use_container_width=True)
        c2.image(mask, caption="Inpainting Mask", use_container_width=True)
        c3.image(restored, caption=f"Restored ({inpaint_method.split()[0]})", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 7
    # -------------------------------------------------------------
    elif "Practical 7:" in selected_exp:
        st.header("Practical 7: Lossless vs Lossy Image Compression")
        st.info("**Aim:** Analyze file size compression ratios using JPEG (Lossy) and PNG (Lossless).")

        jpeg_q = st.slider("JPEG Quality (Lossy)", 5, 100, 30)
        png_lvl = st.slider("PNG Compression Level (Lossless, 0-9)", 0, 9, 9)

        # Encode JPEG
        _, enc_jpg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        jpg_size = len(enc_jpg) / 1024.0

        # Encode PNG
        _, enc_png = cv2.imencode('.png', img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, png_lvl])
        png_size = len(enc_png) / 1024.0

        orig_size = (rows * cols * 3) / 1024.0

        st.table({
            "Metric": ["Raw Uncompressed (KB)", "JPEG Lossy Size (KB)", "PNG Lossless Size (KB)",
                       "JPEG Compression Ratio", "PNG Compression Ratio"],
            "Value": [
                f"{orig_size:.2f} KB",
                f"{jpg_size:.2f} KB (Quality={jpeg_q})",
                f"{png_size:.2f} KB (Level={png_lvl})",
                f"{(orig_size / jpg_size):.2f}:1",
                f"{(orig_size / png_size):.2f}:1"
            ]
        })

        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(cv2.imdecode(enc_jpg, 1), cv2.COLOR_BGR2RGB), caption=f"JPEG Output ({jpg_size:.1f} KB)",
                 use_container_width=True)
        c2.image(cv2.cvtColor(cv2.imdecode(enc_png, 1), cv2.COLOR_BGR2RGB), caption=f"PNG Output ({png_size:.1f} KB)",
                 use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 8
    # -------------------------------------------------------------
    elif "Practical 8:" in selected_exp:
        st.header("Practical 8: Morphological Operations")
        st.info("**Aim:** Perform Erosion, Dilation, Opening, and Closing on binary thresholded images.")

        thresh_val = st.slider("Binarization Threshold", 0, 255, 127)
        _, binary = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
        k_size = st.slider("Structuring Element Kernel Size", 1, 15, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

        m_op = st.selectbox("Operation", ["Erosion", "Dilation", "Opening", "Closing"])
        mapping = {
            "Erosion": cv2.erode(binary, kernel, iterations=1),
            "Dilation": cv2.dilate(binary, kernel, iterations=1),
            "Opening": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
            "Closing": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        }
        res_binary = mapping[m_op]

        # Extract contours as in lab manual
        contours, _ = cv2.findContours(res_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured_img = cv2.cvtColor(res_binary, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(contoured_img, contours, -1, (255, 0, 0), 2)

        c1, c2, c3 = st.columns(3)
        c1.image(binary, caption="Binarized Input", use_container_width=True)
        c2.image(res_binary, caption=f"{m_op} Result", use_container_width=True)
        c3.image(contoured_img, caption=f"Contours Detected: {len(contours)}", use_container_width=True)

    # -------------------------------------------------------------
    # PRACTICAL 9
    # -------------------------------------------------------------
    elif "Practical 9:" in selected_exp:
        st.header("Practical 9: Object Detection via Correlation (Template Matching)")
        st.info("**Aim:** Slide a template over an image to detect matching features using `cv2.matchTemplate`.")

        st.write("Upload a target template snippet (cropped portion of the main image) or crop directly:")
        template_file = st.file_uploader("Upload Target Template Image", type=["jpg", "png"], key="p9_templ")

        if template_file:
            templ_rgb = load_cv_image(template_file)
            templ_gray = cv2.cvtColor(templ_rgb, cv2.COLOR_RGB2GRAY)
            th_h, th_w = templ_gray.shape

            corr_res = cv2.matchTemplate(img_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
            threshold = st.slider("Correlation Threshold", 0.1, 1.0, 0.7)
            loc = np.where(corr_res >= threshold)

            detected_img = img_rgb.copy()
            match_count = 0
            for pt in zip(*loc[::-1]):
                cv2.rectangle(detected_img, pt, (pt[0] + th_w, pt[1] + th_h), (255, 255, 0), 3)
                match_count += 1

            c1, c2 = st.columns(2)
            c1.image(templ_rgb, caption="Template to Find", width=180)
            c2.image(detected_img, caption=f"Detection Results ({match_count} points above threshold)",
                     use_container_width=True)

    # -------------------------------------------------------------
    # POST-LAB 1
    # -------------------------------------------------------------
    elif "Post-Lab 1:" in selected_exp:
        st.header("Post-Lab 1: Frequency Domain Filtering (2D-DFT)")
        st.info(
            "**Aim:** Compute Discrete Fourier Transform magnitude spectrum and apply Ideal Low/High-pass frequency filtering.")

        dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        norm_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        filter_choice = st.radio("Frequency Domain Filter",
                                 ["Magnitude Spectrum Only", "Ideal Low-Pass Filter", "Ideal High-Pass Filter"])

        if filter_choice == "Magnitude Spectrum Only":
            c1, c2 = st.columns(2)
            c1.image(img_gray, caption="Grayscale Spatial Image", use_container_width=True)
            c2.image(norm_mag, caption="2D-DFT Frequency Spectrum", use_container_width=True)
        else:
            radius = st.slider("Cutoff Frequency Radius (D0)", 10, min(rows, cols) // 2, 40)
            crow, ccol = rows // 2, cols // 2
            y, x = np.ogrid[:rows, :cols]
            dist_from_center = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)

            if filter_choice == "Ideal Low-Pass Filter":
                mask = dist_from_center <= radius
            else:
                mask = dist_from_center > radius

            fshift = dft_shift.copy()
            fshift[:, :, 0] *= mask
            fshift[:, :, 1] *= mask

            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)

            c1, c2, c3 = st.columns(3)
            c1.image(img_gray, caption="Original Gray", use_container_width=True)
            c2.image(mask.astype(np.uint8) * 255, caption="Frequency Domain Filter Mask", use_container_width=True)
            c3.image(img_back.astype(np.uint8), caption="Inverse DFT Reconstructed", use_container_width=True)

    # -------------------------------------------------------------
    # POST-LAB 2
    # -------------------------------------------------------------
    elif "Post-Lab 2:" in selected_exp:
        st.header("Post-Lab 2: Advanced Color Space Conversions")
        st.info(
            "**Aim:** Convert images between RGB, HSV, YCrCb, and CIELAB color spaces and analyze individual channels.")

        space = st.selectbox("Select Color Space Model", ["HSV", "YCrCb", "CIELAB", "RGB Channel Split"])

        if space == "HSV":
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            st.image(hsv, caption="Full HSV Representation", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.image(h, caption="Hue Channel (0-179)", use_container_width=True)
            c2.image(s, caption="Saturation Channel (0-255)", use_container_width=True)
            c3.image(v, caption="Value/Brightness (0-255)", use_container_width=True)

        elif space == "YCrCb":
            ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            st.image(ycrcb, caption="Full YCrCb Representation", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.image(y, caption="Y (Luminance)", use_container_width=True)
            c2.image(cr, caption="Cr (Red Chroma Difference)", use_container_width=True)
            c3.image(cb, caption="Cb (Blue Chroma Difference)", use_container_width=True)

        elif space == "CIELAB":
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            st.image(lab, caption="Full CIELAB Representation", use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.image(l, caption="L* (Perceptual Lightness)", use_container_width=True)
            c2.image(a, caption="a* (Green - Magenta Axis)", use_container_width=True)
            c3.image(b, caption="b* (Blue - Yellow Axis)", use_container_width=True)

        else:
            r = img_rgb[:, :, 0]
            g = img_rgb[:, :, 1]
            b = img_rgb[:, :, 2]
            c1, c2, c3 = st.columns(3)
            c1.image(r, caption="Red Channel", use_container_width=True)
            c2.image(g, caption="Green Channel", use_container_width=True)
            c3.image(b, caption="Blue Channel", use_container_width=True)

else:
    st.info("👈 Please upload an image from the left sidebar to start exploring the experiments!")