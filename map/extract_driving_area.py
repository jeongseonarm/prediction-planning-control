import cv2
import numpy as np
import os


# ============================================================
# Configuration
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(
    HERE, "..", "trajectory_prediction", "Data", "inD_backgroud_images", "18_background.png"
)
MASK_PATH = os.path.join(HERE, "drivable_mask_images", "18_mask.png")


# ============================================================
# Global Variables
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

print(image.shape)

mask = None
points = []


# ============================================================
# Mouse Callback
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point: ({x}, {y})")

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        if len(points) > 1:
            cv2.line(image, points[-2], points[-1], (0, 255, 0), 2)

        cv2.imshow("Map Editor", image)


# ============================================================
# Existing Mask
# ============================================================

if os.path.exists(MASK_PATH):
    mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
    mask_vis = mask * 255

    cv2.imshow("Mask", mask_vis)

    print("Press ESC to exit.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cv2.destroyAllWindows()


# ============================================================
# Create New Mask
# ============================================================

else:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    print("Left Click : Add polygon point")
    print("Z          : Undo last point")
    print("ENTER      : Create mask")
    print("R          : Reset")
    print("S          : Save mask")
    print("ESC        : Exit")

    cv2.namedWindow("Map Editor")
    cv2.setMouseCallback("Map Editor", mouse_callback)

    while True:
        cv2.imshow("Map Editor", image)

        key = cv2.waitKey(1) & 0xFF

        # Create mask
        if key == 13:
            if len(points) < 3:
                print("Need at least 3 points.")
                continue

            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 1)

            print("Mask created.")
            print("Mask values:", np.unique(mask))

            mask_vis = mask * 255
            cv2.imshow("Mask", mask_vis)

        # Undo last point
        elif key == ord("z"):
            if points:
                removed_point = points.pop()
                print(f"Removed point: {removed_point}")

                # redraw the background
                image = cv2.imread(IMAGE_PATH)

                # redraw the remaining points and lines
                for i, point in enumerate(points):
                    cv2.circle(
                        image,
                        point,
                        5,
                        (0, 255, 0),
                        -1
                    )

                    if i > 0:
                        cv2.line(
                            image,
                            points[i - 1],
                            points[i],
                            (0, 255, 0),
                            2
                        )

                cv2.imshow("Map Editor", image)

            else:
                print("No points to remove.")

        # Reset
        elif key == ord("r"):
            points.clear()
            mask[:] = 0
            image = cv2.imread(IMAGE_PATH)

            print("Reset.")

            mask_vis = mask * 255
            cv2.imshow("Mask", mask_vis)

        # Save
        elif key == ord("s"):
            success = cv2.imwrite(MASK_PATH, mask)

            if success:
                print(f"Mask saved: {MASK_PATH}")
                print("Saved values:", np.unique(mask))
            else:
                print("Failed to save mask.")

        # Exit
        elif key == 27:
            break

    cv2.destroyAllWindows()