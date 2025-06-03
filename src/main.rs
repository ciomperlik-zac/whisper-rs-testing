mod whisper;
use whisper::SttEngine;
use windows_sys;

fn main() {
    let stt_engine =
        SttEngine::new("C:\\Users\\zciom\\Desktop\\Rust\\assistant\\whisper\\ggml-base.en.bin");

    let output = stt_engine.transcribe_wav("C:\\Users\\zciom\\output.wav");

    println!("{}", output);
}
