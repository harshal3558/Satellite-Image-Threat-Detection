import json
import os

file_path = "satellite-image-threat-detection-enhanced.ipynb"

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source_str = "".join(cell.get("source", []))
        if "def predict_large_image(" in source_str:
            new_source = [
                "import rasterio\n",
                "from rasterio.windows import Window\n",
                "import numpy as np\n",
                "import cv2\n",
                "import torch\n",
                "import torchvision\n",
                "from ultralytics import YOLO\n",
                "\n",
                "def predict_large_image(image_path, model_path, tile_size=800, overlap=100, iou_threshold=0.45):\n",
                "    \"\"\"\n",
                "    Runs YOLOv8 inference on a large satellite image by tiling it and applies NMS.\n",
                "    \n",
                "    Args:\n",
                "        image_path (str): Path to the large .tif image.\n",
                "        model_path (str): Path to the trained YOLO weights (e.g. best.pt).\n",
                "        tile_size (int): Size of the tile for inference.\n",
                "        overlap (int): Overlap between tiles to prevent cutting objects.\n",
                "        iou_threshold (float): Intersection Over Union threshold for NMS.\n",
                "        \n",
                "    Returns:\n",
                "        list: Global bounding boxes [x1, y1, x2, y2, confidence, class_id] after NMS.\n",
                "    \"\"\"\n",
                "    print(f\"Loading model {model_path}...\")\n",
                "    model = YOLO(model_path)\n",
                "    stride = tile_size - overlap\n",
                "    all_boxes = []\n",
                "    all_scores = []\n",
                "    all_classes = []\n",
                "    \n",
                "    print(f\"Processing large image: {image_path}\")\n",
                "    with rasterio.open(image_path) as src:\n",
                "        width, height = src.width, src.height\n",
                "        \n",
                "        for y in range(0, height, stride):\n",
                "            for x in range(0, width, stride):\n",
                "                w = min(tile_size, width - x)\n",
                "                h = min(tile_size, height - y)\n",
                "                \n",
                "                window = Window(x, y, w, h)\n",
                "                img = src.read([1, 2, 3], window=window)\n",
                "                img = np.transpose(img, (1, 2, 0)) # CHW to HWC\n",
                "                \n",
                "                # Avoid passing tiny fragments\n",
                "                if img.shape[0] < 10 or img.shape[1] < 10:\n",
                "                    continue\n",
                "                \n",
                "                # Run inference on the tile\n",
                "                results = model.predict(img, imgsz=tile_size, verbose=False)\n",
                "                \n",
                "                for r in results:\n",
                "                    for box in r.boxes:\n",
                "                        b = box.xyxy[0].cpu().numpy()\n",
                "                        conf = box.conf[0].cpu().numpy()\n",
                "                        cls = box.cls[0].cpu().numpy()\n",
                "                        \n",
                "                        # Map local tile coordinates to global image coordinates\n",
                "                        gx1, gy1 = b[0] + x, b[1] + y\n",
                "                        gx2, gy2 = b[2] + x, b[3] + y\n",
                "                        \n",
                "                        all_boxes.append([gx1, gy1, gx2, gy2])\n",
                "                        all_scores.append(conf)\n",
                "                        all_classes.append(cls)\n",
                "                        \n",
                "    if not all_boxes:\n",
                "        print(\"No objects detected.\")\n",
                "        return []\n",
                "\n",
                "    # --- Apply Global Non-Maximum Suppression (NMS) ---\n",
                "    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)\n",
                "    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)\n",
                "    classes_tensor = torch.tensor(all_classes, dtype=torch.float32)\n",
                "    \n",
                "    # batched_nms applies NMS separately for each class\n",
                "    keep_indices = torchvision.ops.batched_nms(\n",
                "        boxes=boxes_tensor,\n",
                "        scores=scores_tensor,\n",
                "        idxs=classes_tensor,\n",
                "        iou_threshold=iou_threshold\n",
                "    )\n",
                "    \n",
                "    final_predictions = []\n",
                "    for idx in keep_indices:\n",
                "        i = idx.item()\n",
                "        final_predictions.append([\n",
                "            all_boxes[i][0], all_boxes[i][1], all_boxes[i][2], all_boxes[i][3],\n",
                "            all_scores[i].item(), all_classes[i].item()\n",
                "        ])\n",
                "        \n",
                "    print(f\"Completed inference. Found {len(all_boxes)} total objects, reduced to {len(final_predictions)} after NMS.\")\n",
                "    return final_predictions\n",
                "\n",
                "# ==========================================\n",
                "# Example Usage:\n",
                "# ==========================================\n",
                "# sample_image = f'{IMAGE_DIR}/2355.tif'\n",
                "# trained_model = f'{OUTPUT_DIR}/satellite_detector/weights/best.pt'\n",
                "# \n",
                "# if os.path.exists(sample_image) and os.path.exists(trained_model):\n",
                "#     global_boxes = predict_large_image(sample_image, trained_model, iou_threshold=0.45)\n",
                "#     print(f\"First 5 predictions: {global_boxes[:5]}\")\n",
                "# else:\n",
                "#     print(\"Train the model first to generate weights for inference!\")\n"
            ]
            cell["source"] = new_source
            break

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Added NMS to the notebook successfully.")
