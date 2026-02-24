import cv2
import os

class ImageToolkit:

    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found.")
        
    def resize(self, width, height):
        return cv2.resize(self.image, (width, height))
    
    # Convert to Grayscale
    def to_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    
    def crop(self, x1, y1, x2, y2):
        return self.image[y1:y2, x1:x2]
    # image[row, column]
    # = image[y, x]

    def draw_rectangle(self, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
        image_copy = self.image.copy()
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
        return image_copy
    
    def detect_edges(self, low=100, high=200):
        gray = self.to_grayscale()
        return cv2.Canny(gray, low, high)
    
    def save(self, image, output_path):
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    toolkit = ImageToolkit("data/images/test.png")

    resized = toolkit.resize(300, 300)
    print("Resized image shape:", resized.shape)
    gray = toolkit.to_grayscale()
    print("Grayscale image shape:", gray.shape)
    cropped = toolkit.crop(100, 100, 400, 400)
    print("Cropped image shape:", cropped.shape)
    boxed = toolkit.draw_rectangle(100, 100, 400, 400)
    print("Boxed image shape:", boxed.shape)
    edges = toolkit.detect_edges()
    print("Edges image shape:", edges.shape)

    toolkit.save(resized, "data/resized.png")
    toolkit.save(gray, "data/gray.png")
    toolkit.save(cropped, "data/cropped.png")
    toolkit.save(boxed, "data/boxed.png")
    toolkit.save(edges, "data/edges.png")