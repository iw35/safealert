# SafeAlert

SafeAlert is a notebook-based audio classification project that distinguishes **firework** vs **gunshot** sounds and presents a risk-oriented result in a desktop UI.

## What the Project Does

- Extracts audio features from uploaded sound files using:
	- 40 MFCC coefficients (mean-pooled)
	- 1 zero-crossing-rate feature
- Builds a **41-dimensional** feature vector per sample.
- Uses a Keras classifier to output class probabilities:
	- `Firework` (shown as lower risk / safe)
	- `Gunshot` (shown as higher risk / alert)
- Provides a Tkinter UI to:
	- upload audio,
	- display waveform,
	- show class probability bars,
	- play/stop audio,
	- display confidence and status.

## Repository Overview

- `gui.ipynb`: inference UI notebook (model loading + feature extraction + Tkinter app).
- `rnn_model.ipynb`: training/experimentation notebook (data prep, training loops, evaluation, model save/load).
- `model_keras.h5`: trained full model used by the UI.
- `model_weights.weights.h5`: saved model weights.
- `test/`: sample audio files for quick local testing.
- `LICENSE`: MIT License.

## Requirements

Python 3.9+ is recommended.

Install dependencies (minimum set inferred from notebooks):

```bash
pip install numpy matplotlib librosa tensorflow keras scikit-learn pygame jupyter
```

## Quick Start (Inference UI)

1. Ensure `model_keras.h5` exists in the project root.
2. Start Jupyter:

	 ```bash
	 jupyter notebook
	 ```

3. Open `gui.ipynb`.
4. Run notebook cells in order.
5. In the UI:
	 - Click **Upload Audio** and choose a sound file (`.wav`, `.mp3`, `.ogg`).
	 - Click **Run Classification** to get probabilities and risk status.
	 - Optionally use **Play/Stop** to monitor the uploaded audio.

## Training / Re-Training

1. Open `rnn_model.ipynb`.
2. Run preprocessing and training cells in sequence.
3. The notebook saves model artifacts, including final root outputs such as:
	 - `model_keras.h5`
	 - `model_weights.weights.h5`
4. Re-run `gui.ipynb` to use the updated model.

## Notes

- Inference expects the same feature format used in training (`(1, 41)` input shape).
- The current UI maps predicted class index `0` to Firework and `1` to Gunshot.
- Sample files in `test/` are useful for smoke testing the UI pipeline.
