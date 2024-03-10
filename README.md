# Overview

This is a GPT-2-like model trained on approximately 96k characters of text comprised of collected Dr. Seuss stories. The basic version of this model was trained using character-level tokenization and a simple cross-entropy loss function.

This model was built with guidance from and reference to Andrej Karpathy's nanoGPT YouTube lecture and code repository. See the references section below for links to the relevant resources.

# Data

This model was trained on a dataset of over 96k characters comprised of collected Dr. Seuss stories. See `data/dr_seuss.txt` for the full dataset. The dataset contains the text of:
- The Cat in the Hat
- The Cat in the Hat Comes Back
- Green Eggs and Ham
- One Fish, Two Fish, Red Fish, Blue Fish
- Horton Hears a Who
- How the Grinch Stole Christmas
- The Lorax
- Fox in Socks
- And to Think that I Saw It on Mulberry Street
- Horton Hatches the Egg
- If I Ran the Zoo
- Yertle the Turtle
- Oh, the Places You'll Go!
- There's a Wocket in My Pocket
- The Sneetches
- I Can Read with My Eyes Shut

Currently, the training script uses simple character-level tokenization.

# Usage

The model code is in `model.py`. To generate text with the model, call the model's `generate` function

To train the model, simply run `train.py`. The training script uses hydra for config management and Weights & Biases for experiment tracking.

To generate text, call the `GPT` class's `generate` function.

### Sample output

```
"What ZOM!" game you do!
In them in the cat.
"You know what a wall!
I call not like this one at."
Said the cat
With a box time fox.".

"Look at me
Dood. On the pish Shook!
Sow!" said the cat.
"Look at me little book!" he laul!.
"I should not he did not eat bit
One he hoise
As he house
Lave not here to guy out!
Have no fear."

"I can't shat you is not,"
Said the fish...

No he pated out along,
Mon't on elephant!"..
A hatchings,"..
With pegred cerrabd calleds
tarning lives are glant.
```

# Rerferences
1) [Andrej Karpathy's nanoGPT YouTube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6042s)
2) [Andrej Karpathy's nanoGPT repo](https://github.com/karpathy/nanoGPT/tree/master)
