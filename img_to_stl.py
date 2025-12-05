import numpy as np
from stl import mesh
import cv2
from typing import Tuple, Optional
from pathlib import Path
import argparse


class ImageToSTL:
    def __init__(
            self,
            img_path: str,
            output_path: str,
            max_width: int,
            tile_size: int = 180,
            scale_factor: float = 1.0,
            base_height: float = 2.0,
            blur: Optional[Tuple[int, int]] = None,
            invert: bool = False,
            max_height: float = 10.0,
            threshold: Optional[int] = None,
    ):
        self.img_path = Path(img_path)
        self.max_width = max_width
        self.tile_size = tile_size
        self.scale_factor = scale_factor
        self.base_height = base_height
        self.blur = blur
        self.invert = invert
        self.max_height = max_height
        self.threshold = threshold

        # Read image with alpha channel
        print(f"Reading image: {img_path}")

        if not self.img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        self.img = img
        self.tiles = [img]
        self.img_x_dim = img.shape[1]
        self.img_y_dim = img.shape[0]
        self.img_hastransparency = img.shape[-1] == 4
        self.img_isrgb = len(img.shape) == 3

        if self.max_width > 0:
            #We expect to tile the image, hence the results will be stored in a folder
            self.output = Path(output_path)
            self.output.mkdir(parents=True, exist_ok=True)
        else:
            self.output = Path(output_path)

    def process_image(self) -> None:
        try:
            if self.max_width > 0:
                self.resize()
                self.tile()

            if self.img_hastransparency:
                self.transparency()

            if self.img_isrgb:
                self.grayscale()

            if self.threshold is not None:
                self.threshold()

            # Apply blur if specified (after threshold)
            if self.blur is not None:
                self.gaussianblur()

            for index, img in enumerate(self.tiles):
                v = self.vertices(img)
                d = self.mesh(v)
                self.save(d, index)

        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")

    def resize(self):
        print("Resizing image...")
        h, w = self.img.shape[:2]
        aspect_ratio = w / h

        c = 1
        while True:
            c = c + 1
            r = c / aspect_ratio
            if r.is_integer():
                if self.tile_size * c > self.max_width:
                    break
                else:
                    columns = c
                    rows = int(r)
            else:
                continue

        new_width = columns * self.tile_size
        new_height = int(h * (new_width / w))

        if new_width < w:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        self.img = cv2.resize(self.img, (new_width, new_height), interpolation=interpolation)

        self.img_x_dim = self.img.shape[1] / columns
        self.img_y_dim = self.img.shape[0] / rows

        print(f"Image resized from {w}x{h} to {new_width}x{new_height}")
        print(f"{columns}x{rows} -> {columns * rows} tiles")

        filename = self.img_path.parent / f"{self.img_path.stem}_resized{self.img_path.suffix}"
        print(f"Saving resized image to: {filename}")
        cv2.imwrite(filename, self.img)

    def tile(self):
        h, w = self.img.shape[:2]
        self.tiles = []

        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                tile = self.img[y:y + self.tile_size, x:x + self.tile_size]
                tiledir = self.output / "tiles"
                tiledir.mkdir(exist_ok=True)
                filename = tiledir / f"{len(self.tiles)}.png"
                print(f"Saving tile to: {filename}")
                cv2.imwrite(filename, tile)
                self.tiles.append(tile)

        print(f"Generated {len(self.tiles)} tiles")

    def transparency(self) -> None:
        print("Processing transparent PNG...")
        self.tiles = [self._merge_channels(img) for img in self.tiles]

    def _merge_channels(self, img: cv2.Mat) -> cv2.Mat:
        # Split the image into color and alpha channels
        alpha = img[:, :, 3]

        # Create a white background
        white_background = np.ones_like(img[:, :, :3]) * 255

        # Calculate foreground ratio based on alpha
        alpha_3d = np.stack([alpha, alpha, alpha], axis=-1) / 255.0

        # Combine foreground and background
        return (img[:, :, :3] * alpha_3d + white_background * (1 - alpha_3d)).astype(np.uint8)

    def grayscale(self) -> None:
        print("Converting to grayscale...")
        self.tiles = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.tiles]

    def threshold(self) -> None:
        print(f"Applying threshold at level {self.threshold}...")
        self.tiles = [cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)[1] for img in self.tiles]

    def gaussianblur(self) -> None:
        print(f"Applying Gaussian blur with kernel size {self.blur}...")
        self.tiles = [cv2.GaussianBlur(img, self.blur, 0) for img in self.tiles]

    def vertices(self, img: cv2.Mat) -> list:
        # Scale the height values
        height_scale = self.max_height / 255.0

        # Generate 3D mesh
        print("Generating 3D mesh...")
        vertices_array = []
        for x, row in enumerate(img):
            for y, pixel in enumerate(row):
                # Scale the coordinates
                x_scaled = x * self.scale_factor
                y_scaled = y * self.scale_factor

                # Calculate height based on pixel value
                height = (255 - pixel if self.invert else pixel) * height_scale

                # Create vertices for current pixel
                vertices = np.array(
                    [
                        # Base vertices
                        [x_scaled, y_scaled, 0],
                        [x_scaled + self.scale_factor, y_scaled, 0],
                        [
                            x_scaled + self.scale_factor,
                            y_scaled + self.scale_factor,
                            0
                        ],
                        [x_scaled, y_scaled + self.scale_factor, 0],
                        # Top vertices
                        [x_scaled, y_scaled, height + self.base_height],
                        [
                            x_scaled + self.scale_factor,
                            y_scaled,
                            height + self.base_height
                        ],
                        [
                            x_scaled + self.scale_factor,
                            y_scaled + self.scale_factor,
                            height + self.base_height
                        ],
                        [
                            x_scaled,
                            y_scaled + self.scale_factor,
                            height + self.base_height
                        ]
                    ]
                )
                vertices_array.append(vertices)

        if len(vertices_array) == 0:
            raise ValueError(
                "No pixels to extrude! Check your threshold and invert settings."
            )

        return vertices_array

    def mesh(self, arr: list) -> np.ndarray:
        # Define faces for a cube
        faces = np.array(
            [
                [0, 3, 1],
                [1, 3, 2],  # bottom
                [0, 4, 7],
                [0, 7, 3],  # left
                [4, 5, 6],
                [4, 6, 7],  # top
                [5, 1, 2],
                [5, 2, 6],  # right
                [2, 3, 6],
                [3, 7, 6],  # front
                [0, 1, 5],
                [0, 5, 4],  # back
            ]
        )

        # Create meshes
        print("Creating final mesh...")
        meshes = []
        for vertices in arr:
            cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    cube.vectors[i][j] = vertices[f[j], :]
            meshes.append(cube)

        # Combine all meshes
        total_length_data = sum(len(m.data) for m in meshes)
        data = np.zeros(total_length_data, dtype=mesh.Mesh.dtype)
        data["vectors"] = np.array(meshes).reshape((-1, 9)).reshape((-1, 3, 3))

        return data

    def save(self, data: np.ndarray, index: int) -> None:
        # Create and save final mesh
        print(f"Saving STL file to: {self.output}")
        mesh_final = mesh.Mesh(data.copy())

        if self.output.is_dir():
            stldir = self.output / "stl"
            stldir.mkdir(exist_ok=True)
            file = stldir / f"{index}.stl"
        else:
            file = self.output

        mesh_final.save(file)
        print(f"\nSuccess! STL file created: {file}")
        print(f"Model dimensions: {self.img_x_dim * self.scale_factor:.1f}x{self.img_y_dim * self.scale_factor:.1f}x{self.max_height + self.base_height:.1f}mm")

def parse_blur(blur_str: str) -> Tuple[int, int]:
    try:
        w, h = map(int, blur_str.split(","))
        if w % 2 == 0 or h % 2 == 0:
            raise ValueError("Blur kernel dimensions must be odd numbers")
        return (w, h)
    except:
        raise argparse.ArgumentTypeError(
            "Blur must be two odd integers separated by comma (e.g., '5,5')"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image to an STL file for 3D printing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with default settings:
  python image_to_stl.py input.png output.stl

  # Sharp edges with no base layer:
  python image_to_stl.py input.png output.stl --threshold 128 --base-height 0 --invert

  # Set scale factor and maximum height:
  python image_to_stl.py input.png output.stl --scale 0.5 --max-height 20

  # Full example with all options:
  python image_to_stl.py input.png output.stl --scale 0.5 --base-height 0 --max-height 15 --threshold 128 --invert
        """,
    )

    parser.add_argument("input", help="Input image file path")
    parser.add_argument("output", help="Output STL file path")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for X and Y dimensions (default: 1.0)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="If this argument is provided, the image is cropped into tiles. The number of tiles is calculated by considering the given maximum width, the preservation of the aspect ratio of the image, and the tile size."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=180,
        help="Size of the print in mm (default: 180)."
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=2.0,
        help="Height of the base in mm (default: 2.0)",
    )
    parser.add_argument(
        "--max-height",
        type=float,
        default=10.0,
        help="Maximum height in mm (default: 10.0)",
    )
    parser.add_argument(
        "--blur",
        type=parse_blur,
        metavar="W,H",
        help="Apply Gaussian blur with kernel size WxH (e.g., 5,5)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the height mapping (dark pixels become high)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        choices=range(0, 256),
        metavar="[0-255]",
        help="Threshold value for creating sharp edges (0-255)",
    )

    args = parser.parse_args()

    try:
        converter = ImageToSTL(
            img_path=args.input,
            output_path=args.output,
            max_width=args.max_width,
            tile_size=args.tile_size,
            scale_factor=args.scale,
            base_height=args.base_height,
            blur=args.blur,
            invert=args.invert,
            max_height=args.max_height,
            threshold=args.threshold,
        )

        converter.process_image()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
