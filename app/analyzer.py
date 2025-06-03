import io
import math
import fractions
from typing import Dict
from PIL import Image, ImageFilter, ImageStat, ImageCms
import numpy as np
from scipy.ndimage import generic_filter

async def analyze_image(file) -> Dict:
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    width, height = image.size

    # Aspect ratio calculations
    aspect_ratio = width / height if height != 0 else 0
    try:
        simplified = fractions.Fraction(width, height).limit_denominator()
        aspect_ratio_simple = f"{simplified.numerator}:{simplified.denominator}"
    except ZeroDivisionError:
        aspect_ratio_simple = "Undefined"

    # Max print size at 300 DPI
    max_print_width = round(width / 300, 2)
    max_print_height = round(height / 300, 2)
    max_print_width_rounded = math.floor(max_print_width)
    max_print_height_rounded = math.floor(max_print_height)

    # Convert to grayscale for metrics
    image_gray = image.convert("L")

    # Blur variance (using edges)
    laplacian = image_gray.filter(ImageFilter.FIND_EDGES)
    blur_variance = round(ImageStat.Stat(laplacian).var[0], 2)

    # Brightness and contrast (stddev)
    stat = ImageStat.Stat(image_gray)
    brightness = round(stat.mean[0], 2)
    contrast = round(stat.stddev[0], 2)

    # Noise (proxy = contrast)
    noise_level = contrast

    # Dominant color (approximation)
    try:
        small = image.resize((50, 50))
        colors = small.getcolors(2500)
        dominant_color = max(colors, key=lambda item: item[0])[1]
        dominant_color = f"RGB{dominant_color}"
    except Exception:
        dominant_color = "Unavailable"

    # ICC Profile detection
    try:
        icc_profile = image.info.get("icc_profile", None)
        if icc_profile:
            profile = ImageCms.getProfileName(io.BytesIO(icc_profile))
        else:
            profile = "No embedded ICC profile (likely sRGB)"
    except Exception:
        profile = "Unknown or unsupported ICC profile"

    # Closest common aspect ratio + crop suggestion
    common_aspect_ratios = [
        (1, 1), (3, 2), (4, 3), (5, 4), (16, 9),
        (7, 5), (5, 3), (3, 1), (2, 1)
    ]

    def closest_aspect_ratio(target):
        return min(common_aspect_ratios, key=lambda r: abs(r[0] / r[1] - target))

    closest_ratio = closest_aspect_ratio(aspect_ratio)
    closest_ratio_str = f"{closest_ratio[0]}:{closest_ratio[1]}"

    def crop_fit_size(w, h, ratio):
        target_width = h * ratio[0] / ratio[1]
        target_height = w * ratio[1] / ratio[0]
        if target_width <= w:
            return int(target_width), h
        else:
            return w, int(target_height)

    crop_width, crop_height = crop_fit_size(width, height, closest_ratio)
    crop_print_width = round(crop_width / 300, 2)
    crop_print_height = round(crop_height / 300, 2)

    # Score ranges definitions
    blur_range = (1000, 6000)
    noise_range = (0, 50)
    brightness_range = (100, 180)
    contrast_range = (30, 80)

    def score_status(value, value_range):
        low, high = value_range
        if value < low:
            return "Low"
        elif value > high:
            return "High"
        else:
            return "Good"

    blur_score = score_status(blur_variance, blur_range)
    noise_score = score_status(noise_level, noise_range)
    brightness_score = score_status(brightness, brightness_range)
    contrast_score = score_status(contrast, contrast_range)

    # Flaw detection function
    def detect_flaws(image_gray, blur_variance, noise_level, brightness, contrast):
        flaws = []

        BLUR_THRESHOLD_LOW = blur_range[0]
        NOISE_THRESHOLD_HIGH = noise_range[1]
        BRIGHTNESS_LOW = brightness_range[0]
        BRIGHTNESS_HIGH = brightness_range[1]
        CONTRAST_LOW = contrast_range[0]
        CONTRAST_HIGH = contrast_range[1]
        DUST_SPOT_THRESHOLD = 10  # Number of pixels over threshold indicating dust spots

        if blur_variance < BLUR_THRESHOLD_LOW:
            flaws.append("Image appears blurry or out of focus.")

        if noise_level > NOISE_THRESHOLD_HIGH:
            flaws.append("Image has high noise/grain, which may affect print quality.")

        if brightness < BRIGHTNESS_LOW:
            flaws.append("Image appears underexposed or too dark.")

        if brightness > BRIGHTNESS_HIGH:
            flaws.append("Image appears overexposed or too bright.")

        if contrast < CONTRAST_LOW:
            flaws.append("Image has low contrast and may appear flat.")

        if contrast > CONTRAST_HIGH:
            flaws.append("Image has very high contrast; some details may be clipped.")

        # Dust spot detection
        median = image_gray.filter(ImageFilter.MedianFilter(size=5))
        diff = np.array(image_gray).astype(int) - np.array(median).astype(int)
        diff_abs = np.abs(diff)
        spots_mask = diff_abs > 20
        spots_count = np.sum(spots_mask)
        if spots_count > DUST_SPOT_THRESHOLD:
            flaws.append("Possible dust spots detected on the image.")

        # Noise pattern detection (simplified heuristic)
        kernel_size = 7
        img_np = np.array(image_gray).astype(float)

        def local_std(x):
            return np.std(x)

        local_std_dev = generic_filter(img_np, local_std, size=kernel_size)
        local_std_mean = np.mean(local_std_dev)
        if noise_level > NOISE_THRESHOLD_HIGH and local_std_mean > 20:
            flaws.append("Noise pattern detected; image may have grainy or patterned noise.")

        return flaws

    flaws_detected = detect_flaws(image_gray, blur_variance, noise_level, brightness, contrast)

    # Explanation text
    explanation = (
        "Blur variance (edges/sharpness): higher is better. "
        "Noise level (grain): lower is better. "
        "Brightness (0-255): ideal range is 100–180. "
        "Contrast: under 30 may appear flat, over 80 may clip details. "
        "Dominant color impacts overall mood. "
        "You can optionally crop to a common aspect ratio for better print formats."
    )

    # Suggested improvements (Photoshop / Lightroom)
    suggested_improvements = {
        "blur": "Use sharpening filters (Photoshop: Filter > Sharpen > Unsharp Mask). "
                "In Lightroom, increase Clarity and Sharpening sliders.",
        "noise": "Apply noise reduction (Photoshop: Filter > Noise > Reduce Noise). "
                 "In Lightroom, use the Detail panel's Noise Reduction sliders.",
        "brightness": "Adjust exposure or brightness (Photoshop: Image > Adjustments > Brightness/Contrast). "
                      "In Lightroom, adjust Exposure slider.",
        "contrast": "Use contrast adjustment (Photoshop: Image > Adjustments > Brightness/Contrast or Curves). "
                    "In Lightroom, use Contrast and Tone Curve adjustments.",
        "dust_spots": "Use Spot Healing Brush (Photoshop) or Spot Removal tool (Lightroom) to remove dust spots.",
    }

    # Final result dictionary
    result = {
        "filename": file.filename,
        "width_px": width,
        "height_px": height,
        "aspect_ratio_decimal": round(aspect_ratio, 4),
        "aspect_ratio_simple": aspect_ratio_simple,
        "max_print_size_inch": {
            "precise": {
                "width": max_print_width,
                "height": max_print_height,
                "dpi": 300
            },
            "rounded_down": {
                "width": max_print_width_rounded,
                "height": max_print_height_rounded,
                "dpi": 300
            }
        },
        "blur_variance": {
            "value": blur_variance,
            "range": blur_range,
            "status": blur_score,
            "message": f"{blur_variance} / Ideal: {blur_range[0]}–{blur_range[1]} ({blur_score})"
        },
        "noise_level": {
            "value": noise_level,
            "range": noise_range,
            "status": noise_score,
            "message": f"{noise_level} / Ideal: {noise_range[0]}–{noise_range[1]} ({noise_score})"
        },
        "brightness": {
            "value": brightness,
            "range": brightness_range,
            "status": brightness_score,
            "message": f"{brightness} / Ideal: {brightness_range[0]}–{brightness_range[1]} ({brightness_score})"
        },
        "contrast": {
            "value": contrast,
            "range": contrast_range,
            "status": contrast_score,
            "message": f"{contrast} / Ideal: {contrast_range[0]}–{contrast_range[1]} ({contrast_score})"
        },
        "dominant_color": dominant_color,
        "color_profile": profile,
        "closest_common_aspect_ratio": closest_ratio_str,
        "crop_suggestion": {
            "recommended_crop_size_px": {
                "width": crop_width,
                "height": crop_height
            },
            "max_print_size_inch": {
                "width": crop_print_width,
                "height": crop_print_height,
                "dpi": 300
            },
            "prompt": f"Would you like to crop to {closest_ratio_str}?"
        },
        "flaws_detected": flaws_detected,
        "suggested_improvements": suggested_improvements,
        "explanation": explanation,
        "message": "Analysis complete"
    }

    return result




