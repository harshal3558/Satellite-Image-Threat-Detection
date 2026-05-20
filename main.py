import argparse
import sys
import os
from src.SITD.pipelines.training_pipeline import TrainingPipeline
from src.SITD.pipelines.prediction_pipeline import PredictionPipeline
from src.SITD.logger import logging

def run_training(args):
    logging.info("Starting training via CLI.")
    try:
        pipeline = TrainingPipeline(
            raw_images_dir=args.raw_images_dir,
            geojson_path=args.geojson_path,
            output_dir=args.output_dir,
            trained_model_dir=args.trained_model_dir
        )
        best_model_path = pipeline.run_pipeline()
        print(f"Training completed successfully! Model weights saved at: {best_model_path}")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)

def run_prediction(args):
    logging.info("Starting prediction via CLI.")
    try:
        if not os.path.exists(args.model_path):
            print(f"Error: Model weights not found at path {args.model_path}")
            sys.exit(1)
            
        pipeline = PredictionPipeline(model_path=args.model_path)
        predictions = pipeline.predict_large_image(
            image_path=args.image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            iou_threshold=args.iou,
            conf_threshold=args.conf
        )
        
        print(f"\nPrediction completed. Detected {len(predictions)} objects.")
        for idx, pred in enumerate(predictions[:10]):
            print(f"[{idx}] Class: {pred['class_id']} | Conf: {pred['confidence']:.3f} | BBox: {pred['box']}")
            
        if len(predictions) > 10:
            print(f"... and {len(predictions) - 10} more detections.")
            
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Satellite Image Threat Detection Pipeline")
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run the full training pipeline")
    train_parser.add_argument("--raw_images_dir", required=True, help="Directory containing raw training images")
    train_parser.add_argument("--geojson_path", required=True, help="Path to geojson annotations file")
    train_parser.add_argument("--output_dir", default="xview_yolo", help="Directory where processed chips will be saved")
    train_parser.add_argument("--trained_model_dir", default="artifacts/models", help="Directory to save trained model weights")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run sliding-window inference on a large satellite image")
    predict_parser.add_argument("--image", required=True, help="Path to large satellite image (.tif)")
    predict_parser.add_argument("--model_path", required=True, help="Path to trained model weights (.pt)")
    predict_parser.add_argument("--tile_size", type=int, default=800, help="Tile size for sliding window inference")
    predict_parser.add_argument("--overlap", type=int, default=100, help="Overlap pixel count between tiles")
    predict_parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for global NMS")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "predict":
        run_prediction(args)

if __name__ == "__main__":
    main()
