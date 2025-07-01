import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

import os


# Load images
flag_path = "./images/flag.png"
pattern_path = "./images/pattern.png"

print("Current directory:", os.getcwd())
print("Flag exists:", os.path.exists(flag_path))
print("Pattern exists:", os.path.exists(pattern_path))
flag = cv.imread(flag_path)
pattern = cv.imread(pattern_path)
def show_image(title, img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(7, 5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Display initial images
show_image("Original Flag", flag)
show_image("Pattern Image", pattern)

if flag is None or pattern is None:
    raise FileNotFoundError("Could not load input images.")

clicked_points = []
phase = "corners"

# Try to load saved points
if os.path.exists("saved_points.npy"):
    use_saved = input("Reuse previously saved points? (y/n): ").strip().lower()
    if use_saved == 'y':
        clicked_points = np.load("saved_points.npy", allow_pickle=True).tolist()
        if len(clicked_points) < 4:
            raise ValueError("Saved points must include at least 4 corner points.")
        print(f"Loaded {len(clicked_points)} points from 'saved_points.npy'")
    else:
        print("Proceeding with manual point selection")

if not clicked_points:
    def click_callback(event, x, y, flags, param):
        global clicked_points, phase
        if event == cv.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            if phase == "corners" and len(clicked_points) <= 4:
                print(f"Corner {len(clicked_points)}: {x}, {y}")
            elif phase == "mask":
                print(f"Mask Point {len(clicked_points)-4}: {x}, {y}")

    cv.namedWindow("Select Points")
    cv.setMouseCallback("Select Points", click_callback)

    print("First, click 4 corner points in this order:")
    print("Top-left → Top-right → Bottom-right → Bottom-left")
    print("Then press 'n' to continue to mask point selection.")
    print("Press Backspace to undo the last point.\n")

    while True:
        temp = flag.copy()
        for idx, pt in enumerate(clicked_points):
            cv.circle(temp, pt, 4, (0, 0, 255), -1)
            cv.putText(temp, str(idx+1), (pt[0]+5, pt[1]-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.imshow("Select Points", temp)

        key = cv.waitKey(1) & 0xFF
        if key == ord('n'):
            if phase == "corners":
                if len(clicked_points) < 4:
                    print("You need to select all 4 corner points before continuing.")
                else:
                    print("Now select mask boundary points (cloth outline). Press 'n' again when done.")
                    phase = "mask"
            elif phase == "mask":
                if len(clicked_points) < 4:
                    print("You have undone a corner point. Select 4 corners first.")
                    phase = "corners"
                else:
                    break
        elif key == 8:  # Backspace
         if clicked_points:
            removed = clicked_points.pop()
            print(f"Undid point: {removed}")
            if len(clicked_points) < 4:
                phase = "corners"
                print("Less than 4 corner points remaining. Returning to corner selection.")

    cv.destroyAllWindows()
    np.save("saved_points.npy", np.array(clicked_points))
    print(f"Saved {len(clicked_points)} points to 'saved_points.npy'")

corner_pts = np.array(clicked_points[:4], dtype=np.float32)
mask_pts = np.array(clicked_points[4:], dtype=np.int32)

# Warp pattern
h_pat, w_pat = pattern.shape[:2]
src_pts = np.float32([[0, 0], [w_pat, 0], [w_pat, h_pat], [0, h_pat]])
M = cv.getPerspectiveTransform(src_pts, corner_pts)
warped_pattern = cv.warpPerspective(pattern, M, (flag.shape[1], flag.shape[0]))

# Displace pattern
gray = cv.cvtColor(flag, cv.COLOR_BGR2GRAY)
gray_filtered = cv.bilateralFilter(gray, 9, 75, 75)
sobel_x = cv.Sobel(gray_filtered, cv.CV_32F, 1, 0, ksize=5)
sobel_y = cv.Sobel(gray_filtered, cv.CV_32F, 0, 1, ksize=5)
dx = cv.normalize(sobel_x, None, alpha=-10, beta=10, norm_type=cv.NORM_MINMAX)
dy = cv.normalize(sobel_y, None, alpha=-10, beta=10, norm_type=cv.NORM_MINMAX)
h, w = flag.shape[:2]
map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
map_x = map_x.astype(np.float32) + dx
map_y = map_y.astype(np.float32) + dy
displaced_pattern = cv.remap(warped_pattern, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

# Create mask
mask = np.zeros((h, w), dtype=np.uint8)
if len(mask_pts) > 0:
    cv.fillPoly(mask, [mask_pts], 255)

# Lighting
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
heightmap = clahe.apply(gray)
heightmap = cv.GaussianBlur(heightmap, (5, 5), 0).astype(np.float32) / 255.0
sobel_x = cv.Sobel(heightmap, cv.CV_32F, 1, 0, ksize=5)
sobel_y = cv.Sobel(heightmap, cv.CV_32F, 0, 1, ksize=5)
normal_x = -sobel_x
normal_y = -sobel_y
normal_z = np.ones_like(heightmap)
norm = np.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
normal_x /= norm
normal_y /= norm
normal_z /= norm
light_dir = np.array([0.4, 0.4, 1.0])
light_dir /= np.linalg.norm(light_dir)
dot = (normal_x * light_dir[0] + normal_y * light_dir[1] + normal_z * light_dir[2])
dot = np.clip(dot, 0.4, 1.2)

displaced_float = displaced_pattern.astype(np.float32) / 255.0
shaded = np.zeros_like(displaced_float)
for c in range(3):
    shaded[:, :, c] = displaced_float[:, :, c] * dot
shaded = np.clip(shaded * 255, 0, 255).astype(np.uint8)

# Blend with flag
blended = flag.copy()
alpha = 0.7
blended[mask == 255] = cv.addWeighted(
    shaded[mask == 255], alpha,
    flag[mask == 255], 1 - alpha, 0
)
cv.imwrite("sample_Output.jpg", blended)
print("sample_Output.jpg saved.")

# Overlay blend
bump_output = cv.imread("Output2.jpg")
flag_float = flag.astype(np.float32) / 255.0
bump_float = bump_output.astype(np.float32) / 255.0

overlay_result = np.zeros_like(bump_float)
for c in range(3):
    base = flag_float[:, :, c]
    top = bump_float[:, :, c]
    overlay = np.where(
        base < 0.5,
        2 * base * top,
        1 - 2 * (1 - base) * (1 - top)
    )
    overlay_result[:, :, c] = overlay

strength = 0.1
final_float = (1 - strength) * bump_float + strength * overlay_result
final_img = np.clip(final_float * 255, 0, 255).astype(np.uint8)
cv.imwrite("Output_overlay.jpg", final_img)
print("Output_overlay.jpg saved.")

# Soft light
flag_f = flag.astype(np.float32) / 255.0
bump_f = final_img.astype(np.float32) / 255.0
def soft_light_blend(base, top, intensity=1.0):
    result = np.zeros_like(base)
    for c in range(3):
        b = base[:, :, c]
        t = top[:, :, c]
        soft = np.where(
            t < 0.5,
            2 * b * t + b ** 2 * (1 - 2 * t),
            2 * b * (1 - t) + np.sqrt(b) * (2 * t - 1)
        )
        result[:, :, c] = (1 - intensity) * b + intensity * soft
    return np.clip(result, 0, 1)

softlight_img = (soft_light_blend(flag_f, bump_f, 1.0) * 255).astype(np.uint8)
cv.imwrite("Output_softlight.jpg", softlight_img)

gray_flag = cv.cvtColor(flag, cv.COLOR_BGR2GRAY)
ao_map = clahe.apply(gray_flag)
ao_map = cv.GaussianBlur(ao_map, (5, 5), 0).astype(np.float32) / 255.0
ao_result = np.zeros_like(bump_f)
for c in range(3):
    ao_result[:, :, c] = bump_f[:, :, c] * ao_map
ao_result = np.clip(ao_result * 255, 0, 255).astype(np.uint8)
cv.imwrite("Output_ambient_occlusion.jpg", ao_result)

def show_image(title, img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

show_image("Soft Light Result", softlight_img)
show_image("Ambient Occlusion Result", ao_result)

lab = cv.cvtColor(ao_result, cv.COLOR_BGR2LAB)
l, a, b = cv.split(lab)
l_clahe = cv.createCLAHE(clipLimit=0.85, tileGridSize=(8, 8)).apply(l)
lab_clahe = cv.merge((l_clahe, a, b))
contrast_boosted = cv.cvtColor(lab_clahe, cv.COLOR_LAB2BGR)
cv.imwrite("Output.jpg", contrast_boosted)
print("Saved final contrast-enhanced result as 'Output.jpg'")
# Show final result
show_image("Final Result", contrast_boosted)
