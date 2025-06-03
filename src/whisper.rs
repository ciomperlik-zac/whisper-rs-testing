use hound::WavReader;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct SttEngine {
    context: WhisperContext,
}

impl SttEngine {
    pub fn new(model_path: &str) -> Self {
        let context =
            WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
                .expect("Failed to load model");

        SttEngine { context }
    }

    pub fn transcribe_wav(&self, file_path: &str) -> String {
        let reader = WavReader::open(file_path);
        let original_samples: Vec<i16> = reader
            .expect("Failed to initialize wav reader")
            .into_samples::<i16>()
            .map(|x| x.expect("sample"))
            .collect::<Vec<_>>();

        let mut samples = vec![0.0f32; original_samples.len()];
        whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)
            .expect("Failed to convert samples to audio");

        let mut state = self
            .context
            .create_state()
            .expect("Failed to create whisper state");

        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_initial_prompt("experience");
        params.set_n_threads(8);

        state
            .full(params, &samples)
            .expect("failed to convert samples");

        let mut transcribed = String::new();

        let n_segments = state
            .full_n_segments()
            .expect("Failed to get number of whisper segments");
        for i in 0..n_segments {
            let text = state.full_get_segment_text(i).unwrap_or_default();
            transcribed.push_str(&text);
        }

        transcribed
    }
}
