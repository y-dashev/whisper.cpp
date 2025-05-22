require "whisper"

context = Whisper::Context.new("base.en")


loop do
  # 1. Capture or receive N seconds of audio as PCM (e.g., 3 seconds)
  # 2. Save to temp file or buffer
  # 3. Transcribe
  context.transcribe("chunk.wav", 'en') do |text|
    puts text
  end
end
