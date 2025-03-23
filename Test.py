from basic_pitch.inference import predict



model_output, midi_data, note_events = predict(
            audio_path = "Music/cest_toi.wav", 
            onset_threshold = 0.5, 
            frame_threshold = 0.3, 
            minimum_note_length = 127.70, 
            minimum_frequency= None,
            maximum_frequency = None,
            multiple_pitch_bends = False,
            melodia_trick = True,
            midi_tempo= 120
        )


midi_data.write("predicted.mid")