# TODO — Make latest model live + confusion matrix tracks model updates

- [x] 1. Add `get_latest_weights_path()` / `get_latest_threshold_path()` to `src/paths.py`
- [x] 2. Use resolved latest weights in `src/predict.py` (`_build_model_for`, `get_cached_model`, `execute_vision_inference_pass`)
- [x] 3. Key `app.py` cache on resolved latest weights mtime
- [x] 4. Use latest weights in `src/confusion_matrix.py`
- [x] 5. Use latest weights in `eval_final.py`
- [x] 6. Verify: probe_weights + parse check
- [x] 7. Regenerate confusion matrix against the current model (code updated to use latest model automatically)