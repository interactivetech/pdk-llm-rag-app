

def stream(user_prompt):
    runtimeFlag = "cuda:0"
    # system_prompt = 'The conversation between Human and AI assisatance named Gathnex\n'
    # B_INST, E_INST = "[INST]", "[/INST]"

    # prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

    inputs = tokenizer(user_prompt, return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    model.generate(**inputs, streamer=streamer, max_new_tokens=100)