"""
model.py  –  2× per-channel image upscaler (shared between train / infer)

Architecture (per channel):
  Conv2d(1→128, 9×9, reflect-pad)  + LeakyReLU(0.1)
  Conv2d(128→64, 1×1)              + LeakyReLU(0.1)
  Conv2d(64→16,  1×1)              + LeakyReLU(0.1)
  Conv2d(16→4,   1×1)
  PixelShuffle(2)   →  (B, 1, 2H, 2W)

The network is run on one channel at a time.
Outputs are the RESIDUAL above bilinear interpolation, not raw pixel values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The structure of the network is configurable. 

# k: kernel size (single dimension)
# c: number of kernels
# ms: size of middle layers
# nm: number of middle layers
gConfig = {
    #"k": 9, "c": 128, "ms": 64, "nm": 2 # 90KB
    #"k": 9, "c": 48, "ms": 48, "nm": 3 # ~40KB, bad
    #"k": 9, "c": 64, "ms": 48, "nm": 3 # ~52KB, good <- priority -----
    
    #"k": 7, "c": 64, "ms": 32, "nm": 3 # ~30KB, good
    #"k": 7, "c": 32, "ms": 48, "nm": 3 # ~32KB, good
    #"k": 7, "c": 64, "ms": 48, "nm": 2 # ~35KB, good
    #"k": 7, "c": 32, "ms": 32, "nm": 3 # ~20KB, good <- priority -----
    
    #"k": 5, "c": 32, "ms": 24, "nm": 2 # ~9KB, good
    #"k": 5, "c": 24, "ms": 24, "nm": 2 # ~9KB, good
    #"k": 5, "c": 16, "ms": 24, "nm": 3 # ~9KB, good
    #"k": 5, "c": 24, "ms": 28, "nm": 3 # ~13KB, good <- priority -----
    
    #"k": 5, "c": 16, "ms": 20, "nm": 3 # ~7KB, good <- priority -----
    
    #"k": 5, "c": 12, "ms": 14, "nm": 2 # 3.56KB, bad (checkerboards
    #"k": 5, "c": 12, "ms": 16, "nm": 2 # ~4KB, ...
    #"k": 5, "c": 12, "ms": 16, "nm": 2 # 3.9KB, decent
    
    #"k": 3, "c": 16, "ms": 24, "nm": 3 # ~8KB, good
    #"k": 3, "c": 20, "ms": 20, "nm": 2 # ~5KB, OKish
    #"k": 3, "c": 16, "ms": 16, "nm": 1 # ~2.5KB, ???
    #"k": 3, "c": 16, "ms": 8, "nm": 2 # ~2.1KB, bad
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # ~5KB, OKish
    #"k": 3, "c": 16, "ms": 12, "nm": 2 # ~2.8KB, barely OKish
    #"k": 3, "c": 8, "ms": 16, "nm": 2 # ~2.77KB, OKish somehow
    #"k": 3, "c": 12, "ms": 14, "nm": 2 # ~???KB, barely OKish
    #"k": 3, "c": 12, "ms": 16, "nm": 2 # ~3.2KB, good
    #"k": 5, "c": 10, "ms": 8, "nm": 2 # 2.35KB, bad
    #"k": 5, "c": 16, "ms": 16, "nm": 2 # bad
    #"k": 5, "c": 12, "ms": 12, "nm": 2 # 3.2KB, good <- priority ----- (this is where things start to get dicey for realtime)
    
    #"k": 3, "c": 32, "ms": 32, "nm": 3 # really good actually!
    #"k": 3, "c": 32, "ms": 10, "nm": 3 # disappointing
    #"k": 3, "c": 32, "ms": 16, "nm": 3 # disappointing
    #"k": 3, "c": 32, "ms": 24, "nm": 3 # plausible, cutting early to test 20
    #"k": 3, "c": 32, "ms": 20, "nm": 3 # works but took until epoch 200ish to eliminate all basic artifacts
    
    #"k": 3, "c": 32, "ms": 24, "nm": 2 # plausible at epoch 100
    #"k": 3, "c": 24, "ms": 24, "nm": 2 # implausible
    #"k": 3, "c": 28, "ms": 24, "nm": 2 # blocky gradients at epoch 240 still
    
    #"k": 3, "c": 30, "ms": 22, "nm": 2 # bad
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # artifacty at 200
    #"k": 3, "c": 16, "ms": 24, "nm": 2 # right on the edge of "barely good enough"
    #"k": 3, "c": 16, "ms": 20, "nm": 2 # likewise, but smaller so lol
    #"k": 3, "c": 24, "ms": 20, "nm": 2 # works well, but just barely
    #"k": 3, "c": 20, "ms": 20, "nm": 2 # works well, but just barely
    #"k": 3, "c": 16, "ms": 20, "nm": 3 # low artifact density
    
    #"k": 3, "c": 16, "ms": 20, "nm": 1
    
    #"k": 3, "c": 12, "ms": 10, "nm": 2 # 2.1KBish
    #"k": 3, "c": 10, "ms": 12, "nm": 2 # 2.28KBish
    #"k": 3, "c": 10, "ms": 10, "nm": 2 # 1.99KBish
    
    #"k": 3, "c": 12, "ms": 12, "nm": 3 # 3.21KB, good (yes)
    
    #"k": 5, "c": 12, "ms": 8, "nm": 2 # ?
    #"k": 5, "c": 12, "ms": 8, "nm": 1 # ? failed to train lol
    #"k": 5, "c": 16, "ms": 16, "nm": 1 # ? 3.4KB, looks good.
    #"k": 5, "c": 12, "ms": 16, "nm": 1 # ? 2.7KB, worse than below.
    #"k": 3, "c": 12, "ms": 10, "nm": 2 # ...., looks good. consider it.
    #"k": 3, "c": 12, "ms": 6, "nm": 2 # 1.6KB -- failure to converge
    
    #"k": 3, "c": 10, "ms": 12, "nm": 1 # ~1.5KB, bad
    #"k": 3, "c": 12, "ms": 6, "nm": 2 # ~1.6KB, bad
    #"k": 3, "c": 12, "ms": 8, "nm": 2 # ~1.9KB, good <- priority ----- (this is where realtime becomes viable)
    #"k": 5, "c": 10, "ms": 6, "nm": 2 # ~2.1KB, bad
    #"k": 5, "c": 10, "ms": 12, "nm": 1 # ~2.2KB, ...ehhh, worse than 3_12_8_2 so far
    
    
    #"k": 3, "c": 16, "ms": 8, "nm": 0 # ~1.1KB, bad
    #"k": 3, "c": 12, "ms": 8, "nm": 1 # ~1.4KB, ehhhh
    #"k": 3, "c": 8, "ms": 8, "nm": 2 # ~1.57KB, ehhhh
    #"k": 3, "c": 10, "ms": 10, "nm": 1 # 1.42KB, best of this lot, but...
    #"k": 3, "c": 8, "ms": 12, "nm": 1 # 1.35KB, bad
    #"k": 3, "c": 12, "ms": 12, "nm": 1 # 1.71KB... no, cancel
    #"k": 3, "c": 12, "ms": 10, "nm": 1 # 1.52KB... good enough <- priority ----- (cheapest version that gives reasonable results so far)
    
    #"k": 3, "c": 12, "ms": 0, "nm": 0 # ehhh. it did SOME stuff, but not much.
    #"k": 3, "c": 10, "ms": 6, "nm": 1 # actually works lol <- priority ------
    
    #"k": 3, "c": 13, "ms": 14, "nm": 2 #
    #"k": 3, "c": 9, "ms": 16, "nm": 2 #
    #"k": 3, "c": 14, "ms": 14, "nm": 2
    #"k": 3, "c": 9, "ms": 14, "nm": 1 # bad
    #"k": 3, "c": 9, "ms": 9, "nm": 2 # bad
    #"k": 3, "c": 14, "ms": 10, "nm": 2 # OKish but....
    #"k": 3, "c": 14, "ms": 14, "nm": 1 # very initialization rng sensitive
    #"k": 3, "c": 12, "ms": 16, "nm": 1 # 
    #"k": 3, "c": 9, "ms": 14, "nm": 1 # OK for the size
    #"k": 3, "c": 9, "ms": 9, "nm": 1 #OKish but...
    #"k": 3, "c": 11, "ms": 11, "nm": 1 <--- picked.
    
    
    # WITH NEW: NEW MEANS THAT THERE IS AN ADDITIONAL 4-OUTPUT 3x3 PRE-PASS CONVOLUTION
    
    
    #"new": True, "k": 3, "c": 8, "ms": 12, "nm": 2 # ?
    #"new": True, "k": 3, "c": 11, "ms": 11, "nm": 2 # OK. 759 params. can i get smaller?
    #"new": True, "k": 3, "c": 9, "ms": 11, "nm": 2 # ehhh. (663 params)
    #"new": True, "k": 3, "c": 11, "ms": 9, "nm": 2 # (685) how do?
    #"new": True, "k": 3, "c": 14, "ms": 14, "nm": 2 # good! but 1040 parameters.
    #"new": True, "k": 3, "c": 13, "ms": 11, "nm": 2 # artifacty. 850ish params.
    #"new": True, "k": 3, "c": 12, "ms": 12, "nm": 2 # 848 params. sharp but artifacty.
    #"new": True, "k": 3, "c": 12, "ms": 14, "nm": 2 # already half-decent at 80 epochs! so is 14 outputs the key?
    #"new": True, "k": 3, "c": 8, "ms": 13, "nm": 2 # stuck with jaggies
    #"new": True, "k": 3, "c": 8, "ms": 14, "nm": 2 # already good at 80 epochs lmao
    #"new": True, "k": 3, "c": 8, "ms": 14, "nm": 2 # good at 222 epochs, but... iffy. (732 parameters)
    #"new": True, "k": 3, "c": 9, "ms": 13, "nm": 2 # sus at 80 epochs. 14 in mid layers is the necessity.
    
    #"new": True, "k": 3, "c": 9, "ms": 14, "nm": 2 # good, somehow! but vulnerable to bad initialization. 783 params. see upscaler_n_3x3_9_14_2.safetensors
    
    #"k": 3, "c": 9, "ms": 14, "nm": 2 # 500 params. artifacty at 160 in ways that look like it won't train out.
    #"k": 3, "c": 10, "ms": 14, "nm": 2 # 526 params. artifacty.
    #"k": 3, "c": 12, "ms": 14, "nm": 2 # 526 params. artifacty.
    
    
    
    #"new": True, "k": 3, "c": 16, "ms": 16, "nm": 2 # ew
    #"new": True, "k": 3, "c": 12, "ms": 18, "nm": 2 # yay (very good at 200 epochs)
    #"new": True, "k": 3, "c": 16, "ms": 20, "nm": 2 #
    #"new": True, "k": 3, "c": 14, "ms": 18, "nm": 3 #
    #"new": True, "k": 3, "c": 20, "ms": 22, "nm": 2
    #"new": True, "k": 3, "c": 18, "ms": 16, "nm": 3 # blocky
    #"new": True, "k": 3, "c": 18, "ms": 18, "nm": 3 # blocky
    #"new": True, "k": 3, "c": 20, "ms": 22, "nm": 2 # badly blocky at 450 epoche
    #"new": True, "k": 3, "c": 16, "ms": 18, "nm": 3 # at 250 epochs, not blocky.
    
    #"new": True, "k": 5, "c": 14, "ms": 10, "nm": 2 # too slow
    #"new": True, "k": 5, "c": 11, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 11, "ms": 16, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 12, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 10, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 14, "ms": 10, "nm": 2 # OFF A CLIFF.
    #"new": True, "k": 5, "c": 13, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 32, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 36, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 40, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 48, "nm": 2 # not too slow.
    #"new": True, "k": 5, "c": 13, "ms": 64, "nm": 2 # WAY TOO SLOW
    
    #"new": True, "k": 3, "c": 24, "ms": 24, "nm": 3
    
    #"new": True, "k": 5, "c": 12, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 10, "ms": 24, "nm": 2 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 24, "nm": 3 # not too slow
    #"new": True, "k": 5, "c": 13, "ms": 32, "nm": 2 # not too slow
    
    #"new": True, "k": 3, "c": 32, "ms": 32, "nm": 3 # good, BUT, over-wrought, lmao
    #"new": True, "k": 3, "c": 24, "ms": 28, "nm": 3 #
    #"new": True, "k": 3, "c": 22, "ms": 24, "nm": 3 # wheee (priority ------)
    
    
    # OOPS I IMPLEMENTED "NEW" WRONG LOL AND GROUPS WERE BEING BROADCAST AS CHANNELS. TIME TO REDO IT ALL LOL
    
    
    #"new": True, "k": 3, "c": 24, "ms": 12, "nm": 2 # ...
    #"new": True, "k": 3, "c": 32, "ms": 18, "nm": 1 # good!!! see upscaler - Copy (27)
    #"new": True, "k": 3, "c": 28, "ms": 16, "nm": 1 # dead almost every time. when NOT dead, tends to develop noise later on.
    #"new": True, "k": 3, "c": 24, "ms": 18, "nm": 1 # need good initialization RNG, barely works.
    #"new": True, "k": 3, "c": 28, "ms": 18, "nm": 1 # works more than "barely" but needs good rng
    
    #"new": True, "k": 3, "c": 32, "ms": 12, "nm": 1 # actually works but sometimes has screwy rng. same with 32 -> 14
    #"new": True, "k": 3, "c": 32, "ms": 16, "nm": 1 # actually works and managed to train properly
    #"new": True, "k": 3, "c": 32, "ms": 22, "nm": 2 # -------
    #"new": True, "k": 5, "c": 32, "ms": 26, "nm": 3 # -------
    
    
    
    #"k": 3, "c": 16, "ms": 16, "nm": 2
   
   
   
    #"new": True, "k": 5, "c": 64, "ms": 48, "nm": 3 # overkill
    #"new": True, "k": 5, "c": 96, "ms": 26, "nm": 3 # overkill
    #"new": True, "k": 5, "c": 80, "ms": 32, "nm": 3 # overkill
    #"new": True, "k": 5, "c": 64, "ms": 32, "nm": 3 # overkill
    #"new": True, "k": 5, "c": 72, "ms": 48, "nm": 3 # overkill
    
    #"k": 5, "c": 64, "ms": 48, "nm": 4 # overkill
    
    #"new": True, "k": 5, "c": 40, "ms": 20, "nm": 4 # works if initialized well
    
    #"new": True, "k": 5, "c": 48, "ms": 24, "nm": 3 #  works but bigger than i want to roll with
    #"new": True, "k": 5, "c": 36, "ms": 20, "nm": 3 # not training properly
    #"new": True, "k": 5, "c": 40, "ms": 18, "nm": 3 # training better lol
    #"new": True, "k": 5, "c": 44, "ms": 20, "nm": 2
    
    #"new": True, "k": 5, "c": 44, "ms": 20, "nm": 3
    
    #"new": True, "k": 5, "c": 48, "ms": 20, "nm": 3 # starts good
    #"new": True, "k": 5, "c": 52, "ms": 18, "nm": 3 # improves faster than the above
    #"new": True, "k": 5, "c": 56, "ms": 16, "nm": 3 # keeps radically regressing by epoch 200.
    #"new": True, "k": 5, "c": 56, "ms": 22, "nm": 3 # keeps radically regressing too. maybe on the previous one i just got bad rng?
    #"new": True, "k": 5, "c": 56, "ms": 16, "nm": 3 # trying again
    
    #"new": True, "k": 3, "c": 44, "ms": 16, "nm": 3 # trying again
    
    #"new": True, "k": 3, "c": 24, "ms": 12, "nm": 2
    
    
    #"new": True, "k": 3, "c": 48, "ms": 20, "nm": 3 # actually reasonable lol
    
    #"new": True, "k": 5, "c": 92, "ms": 40, "nm": 4 # "offline"
    
    #"new": True, "k": 3, "c": 16, "ms": 11, "nm": 1 # "almost ultrafast" -- still better than FSR1
    #"new": True, "k": 3, "c": 8, "ms": 8, "nm": 1 # gets stuck easily
    #"new": True, "k": 3, "c": 8, "ms": 11, "nm": 1 # 267 params
    #"new": True, "k": 3, "c": 8, "ms": 12, "nm": 1 # 280 params. better than FSR1, but, REALLY crunchy. gets stuck easily.
    #"new": True, "k": 3, "c": 12, "ms": 8, "nm": 1 # 300 params.
    
    #"new": True, "k": 3, "c": 12, "ms": 7, "nm": 2 # 308 params.
    
    #"new": True, "k": 3, "c": 12, "ms": 7, "nm": 2 # 339 params.
    #"new": True, "k": 3, "c": 16, "ms": 5, "nm": 2 # 372 params.
    
    #"new": True, "k": 3, "c": 8, "ms": 32, "nm": 4 # ok let's just see if 8 filters is ever enough. the answer is yes.
    #"new": True, "k": 3, "c": 4, "ms": 32, "nm": 4 # what about 4 filters? if initialization is good, Yes, somehow.
    
    #"new": True, "k": 3, "c": 8, "ms": 11, "nm": 2 # 399 params. works if init isn't too bad.
    
    #"new": True, "k": 3, "c": 4, "ms": 8, "nm": 1 # 156 params. fails
    #"new": True, "k": 3, "c": 4, "ms": 4, "nm": 3 # 160 params. fails
    #"new": True, "k": 3, "c": 8, "ms": 4, "nm": 2 # 195 params. managed to Start learning. just start, though. it died.
    #"new": True, "k": 3, "c": 8, "ms": 4, "nm": 6 # sanity test. failed to train too.
    #"new": True, "k": 3, "c": 8, "ms": 6, "nm": 6 # learned up-down but not left-right
    #"new": True, "k": 3, "c": 8, "ms": 8, "nm": 2 # 300 params sob
    #"new": True, "k": 3, "c": 8, "ms": 7, "nm": 2 # 271 params
    #"new": True, "k": 3, "c": 8, "ms": 6, "nm": 2
    
    #"new": True, "k": 3, "c": 4, "ms": 32, "nm": 4 # what about 4 filters? ok with bconv-nograd it works (barely)
    #"new": True, "k": 3, "c": 64, "ms": 4, "nm": 8 # shitton of filters, but no feature space?
    #"new": True, "k": 3, "c": 8, "ms": 6, "nm": 4 # 328 params. seems to find a way.
    #"new": True, "k": 3, "c": 8, "ms": 8, "nm": 1 # got stuck
    #"new": True, "k": 3, "c": 8, "ms": 6, "nm": 2 # 244 params. WORKS.
    #"k": 3, "c": 10, "ms": 6, "nm": 2 # 236 params. never learns
    #"k": 3, "c": 8, "ms": 7, "nm": 2 # 231 params. trained up okayishly
    
    #"k": 3, "c": 7, "ms": 6, "nm": 2 # 188 params
    #"k": 3, "c": 7, "ms": 7, "nm": 2 # 214 params
    #"k": 3, "c": 6, "ms": 6, "nm": 3 # 214 params (yes really). trained up well by doing multiple resumptions
    #"k": 3, "c": 10, "ms": 8, "nm": 2 # 296 params
    #"k": 3, "c": 8, "ms": 6, "nm": 2 # 204 params. works for EDI.
    
    #"k": 3, "c": 6, "ms": 4, "nm": 2 # exactly 128 params. fails to train. not enough degrees of freedom before final.
    #"k": 3, "c": 6, "ms": 6, "nm": 1 # 130 params. fails to train up even with EDI.
    #"k": 3, "c": 7, "ms": 5, "nm": 2 # 164 params. fails to train up even with EDI.
    #"k": 3, "c": 8, "ms": 6, "nm": 1 # 162 params. fails to fully train up.
    #"k": 3, "c": 7, "ms": 7, "nm": 1 # 158 params. fails to train up.
    #"k": 3, "c": 6, "ms": 6, "nm": 2 # 172 params. BARELY trains up. at the LAST POSSIBLE minute. and still looks pixelated.
    #"k": 3, "c": 8, "ms": 8, "nm": 1 # 188 params
    #"k": 3, "c": 8, "ms": 8, "nm": 2 # 260 params. works with EDI
    
    #"k": 3, "c": 9, "ms": 8, "nm": 1 # 206 params. trains up on EDI.
    
    #"k": 3, "c": 12, "ms": 7, "nm": 2 # 299 params
    #"k": 3, "c": 14, "ms": 10, "nm": 2 # 444 params
    #"k": 3, "c": 12, "ms": 12, "nm": 2 # 484 params
    #"k": 5, "c": 16, "ms": 8, "nm": 2 # ... params
    #"k": 5, "c": 14, "ms": 12, "nm": 2 # ... params
    #"k": 5, "c": 12, "ms": 11, "nm": 2 # 635 params, bad quality
    #"k": 5, "c": 14, "ms": 12, "nm": 1 # 596 params, bad
    
    
    # ----------
    # THE SHARP EDI ARC
    # ----------
    
    #"k": 3, "c": 9, "ms": 8, "nm": 1 # 206 params. trains up on EDI.
    #"k": 3, "c": 6, "ms": 4, "nm": 1 # 108 params. SORTA works but not really.
    #"k": 3, "c": 4, "ms": 6, "nm": 1 # 98 params. insufficient.
    #"k": 3, "c": 6, "ms": 6, "nm": 1 # 130 params. struggles to not get stuck with a bunch of bad kernels.
    #"k": 3, "c": 8, "ms": 4, "nm": 1 # 136 params. same.
    #"k": 3, "c": 8, "ms": 6, "nm": 1 # 162 params.
    
    #"k": 3, "c": 12, "ms": 8, "nm": 2 # 332 params.
    #"new": True, "k": 3, "c": 12, "ms": 8, "nm": 2 # 332 params.
    
    #"new": True, "k": 7, "c": 64, "ms": 64, "nm": 4 # overkill time! ~90kb
    #"new": True, "k": 5, "c": 36, "ms": 24, "nm": 2 # ~11kb, 37.3 psnr
    #"new": True, "k": 5, "c": 36, "ms": 20, "nm": 3 # ~11kb, ~37.5 psnr
    #"new": True, "k": 5, "c": 40, "ms": 24, "nm": 3 # ~14kb, ~37.7 psnr
    
    #"new": True, "k": 5, "c": 48, "ms": 40, "nm": 4 # ... 38.3 psnr at epoch 270 without bconv nograd. let's try with. ~38.2. ok, so we want without.
    #"new": True, "k": 5, "c": 32, "ms": 40, "nm": 4 # 38.0 psnr at 270 epochs lol
    #"new": True, "k": 5, "c": 40, "ms": 32, "nm": 4 # stuck at 37.5 psnr
    #"new": True, "k": 5, "c": 32, "ms": 48, "nm": 4 # 38.0 psnr at 190 epochs. same at 270 epochs.
    
    #"new": True, "k": 7, "c": 64, "ms": 64, "nm": 5 # 38.6 psnr at 110 epochs. 38.75 at 180 epochs. 
    #"k": 7, "c": 64, "ms": 64, "nm": 5 # 37.2 at 220. 37.4 at 270. 37.9 at 350. 38.2 at 540 epochs. plateau.
    #"new": True, "k": 7, "c": 32, "ms": 64, "nm": 5 # 38.2 at 210.
    #"new": True, "k": 7, "c": 64, "ms": 32, "nm": 5 # 38.1 at 240
    #"new": True, "k": 7, "c": 64, "ms": 64, "nm": 3 # 38.1 at 45 epochs lmao. 38.4 at 95, 38.5~38.6 at 110.
    #"new": True, "k": 7, "c": 48, "ms": 48, "nm": 4 # sucky
    #"new": True, "k": 7, "c": 64, "ms": 48, "nm": 4 # 38.4db at 150. 38.5db at 270.
    #"new": True, "k": 7, "c": 64, "ms": 48, "nm": 2 # 38.2 at 140, gets stuck
    
    #"new": True, "k": 5, "c": 48, "ms": 32, "nm": 3
    #"new": True, "k": 5, "c": 32, "ms": 28, "nm": 3 # 37.5db at 200 epochs. 3500ish params
    #"new": True, "k": 5, "c": 24, "ms": 30, "nm": 2 # 2468 params. 36.7 at 70
    #"new": True, "k": 5, "c": 24, "ms": 30, "nm": 3 # 3392 params. 36.7 at 50. 37.0 at 70. 
    #"new": True, "k": 5, "c": 32, "ms": 24, "nm": 3 # 2900 params. 36.7 at 50. 36.7 at 70.
    #"new": True, "k": 5, "c": 24, "ms": 32, "nm": 2 # 2652 params. stuck at 37.4ish
    
    #"new": True, "k": 5, "c": 32, "ms": 32, "nm": 2 # 3116 params. 37.4db psnr at 150. stuck at ~37.6
    #"new": True, "k": 5, "c": 20, "ms": 40, "nm": 2 # 3204 params. 37.5 at 180. 37.7 at 270. finish at around 37.7
    
    #"new": True, "k": 5, "c": 64, "ms": 48, "nm": 3 # 9724 params. 38.3 psnr
    #"new": True, "k": 7, "c": 72, "ms": 64, "nm": 4 # 21k params. 38.7+ psnr
    #"new": True, "k": 5, "c": 128, "ms": 72, "nm": 3 # ... 38.6ish psnr
    
    ################
    # retraining with augmentation brightness modulation fixed
    ################
    
    #"new": True, "k": 5, "c": 20, "ms": 40, "nm": 2 # 3204 params. 37.7 psnr
    "new": True, "k": 7, "c": 64, "ms": 64, "nm": 4 # 20k params. 38.7 psnr at 120 epochs. 39 at full training. a bit
}

# Intentionally extremely barely-leaky slope so that inference can treat the layer as ReLU instead of LeakyReLU.
# If this causes checkerboard/scanline artifacts in a given model, run a fine-tuning run with train.py --resume
#  and a low learning rate (e.g. 0.0005) and --leaky-slope 0. Doing so will
#  fine-tune with true ReLU and get rid of the artifacts.
LEAKY_SLOPE = 0.0002

class UpscaleNet(nn.Module):
    """
    Input : (B, 1, H, W)  – single channel LR, float in [0, 1]
    Output: (B, 1, 2H, 2W) – residual above bilinear-interpolated input
    """

    def __init__(self, is_wrapping=False):
        super().__init__()
        self.is_wrapping = is_wrapping
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        self._build_layers("new" in gConfig, gConfig["c"], gConfig["ms"], gConfig["nm"], gConfig["k"])
        self._init_weights()

    @staticmethod
    def _config_from_state(state: dict) -> dict:
        """Recover k / c / ms / nm entirely from safetensors weight shapes."""
        conv_w = state["conv.weight"]          # (c, 1, k, k)
        c = conv_w.shape[0]
        k = conv_w.shape[2]

        mid_keys = sorted(
            key for key in state if key.startswith("mids.") and key.endswith(".weight")
        )
        nm = len(mid_keys)
        if nm > 0:
            ms = state[mid_keys[0]].shape[0]  # output channels of mids.0
        else:
            ms = c  # unused when nm == 0, but keep it sane

        ret = {"k": k, "c": c, "ms": ms, "nm": nm}
        
        if "bconv.weight" in state:
            ret["new"] = True
        
        return ret

    def _build_layers(self, new: bool, c: int, ms: int, nm: int, k: int):
        """(Re-)construct all learnable layers from explicit dimensions."""
        self.conv_output_size = c
        self.mid_size         = ms
        self.num_mids         = nm
        self.mids             = nn.ModuleList()
        
        _ichannels = 1
        _igroups = 1
        
        # WITH NEW: NEW MEANS THAT THERE IS AN ADDITIONAL 4-OUTPUT 3x3 PRE-PASS CONVOLUTION
        if new:
            _ichannels = 4
            _igroups = 4
            self.bconv = nn.Conv2d(
                1, _ichannels, kernel_size=3, padding=1,
                padding_mode="circular" if self.is_wrapping else "replicate",
            )
        else:
            try:
                del self.bconv
            except:
                pass
        
        self.conv = nn.Conv2d(
            _ichannels, c, kernel_size=k, padding=(k - 1) // 2, groups=_igroups,
            padding_mode="circular" if self.is_wrapping else "replicate",
        )
        if nm == 0:
            self.zfinal = nn.Conv2d(c, 4, kernel_size=1)
        else:
            self.mids.append(nn.Conv2d(c, ms, kernel_size=1))
            for _ in range(1, nm):
                self.mids.append(nn.Conv2d(ms, ms, kernel_size=1))
            self.zfinal = nn.Conv2d(ms, 4, kernel_size=1)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Infer architecture from *state_dict* shapes, rebuild layers, then load."""
        cfg = self._config_from_state(state_dict)
        
        gConfig = cfg
        self.act = nn.LeakyReLU(LEAKY_SLOPE, inplace=True)
        
        self._build_layers("new" in cfg, cfg["c"], cfg["ms"], cfg["nm"], cfg["k"])
        self._init_weights()
        
        #return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict)

    def _init_weights(self):
        def center_weights_(module):
            if hasattr(module, 'weight'):
                w = module.weight.data
                num_kernels = w.shape[0]
                w_flat = w.view(num_kernels, -1)
                kernel_means = w_flat.mean(dim=1, keepdim=True)
                w_flat.sub_(kernel_means)
                module.weight.data = w_flat.view_as(w)

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(
                nn.init.kaiming_uniform_(
                    m.weight, a=LEAKY_SLOPE, mode="fan_in", nonlinearity="leaky_relu"
                )
                center_weights_(m)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # prepass kernels
            if name == "bconv":
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                with torch.no_grad():
                    ## quiet down the corners
                    #for i in range(0, 4):
                    #    m.weight[i][0][0][0] *= 0.35
                    #    m.weight[i][0][0][2] *= 0.35
                    #    m.weight[i][0][2][0] *= 0.35
                    #    m.weight[i][0][2][2] *= 0.35
                    
                    # quiet down Everything
                    for i in range(0, 4):
                        for y in range(0, 3):
                            for x in range(0, 3):
                                m.weight[i][0][x][y] *= 0.35
                                m.weight[i][0][x][y] *= 0.35
                                m.weight[i][0][x][y] *= 0.35
                                m.weight[i][0][x][y] *= 0.35
                    
                    # force initial axial visibility
                    m.weight[0][0][0][1] = -0.9
                    m.weight[1][0][1][0] = -0.9
                    m.weight[2][0][2][1] = -0.9
                    m.weight[3][0][1][2] = -0.9
                    m.weight[0][0][1][1] = 0.9
                    m.weight[1][0][1][1] = 0.9
                    m.weight[2][0][1][1] = 0.9
                    m.weight[3][0][1][1] = 0.9
                    # if the model really wants to learn other features, it eventually will, but these
                    # first-pass features tend to do the best for the first 20% of training
                    
                # to discourage the network from wanting the prepass kernels to have a negative/compressed output
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W)  →  (B, 1, 2H, 2W) residual"""
        
        # NEW
        if hasattr(self, "bconv"):
            x = self.act(self.bconv(x))
        
        x = self.act(self.conv(x))   # KERNEL_SIZE x KERNEL_SIZE feature extraction
        for i in range(0, self.num_mids):
            x = self.act(self.mids[i](x))
        x = self.zfinal(x)            # no activation on output
        return F.pixel_shuffle(x, 2) # (B,4,H,W) → (B,1,2H,2W)

    @torch.no_grad()
    def upscale_channel(self, lr: torch.Tensor) -> torch.Tensor:
        """
        Upscale one image channel.
        lr : (H, W) float32 tensor in [0, 1]
        returns : (2H, 2W) float32 tensor clamped to [0, 1]
        """
        x        = lr.unsqueeze(0).unsqueeze(0)           # (1,1,H,W)
        base     = upscale_edi_2x(x, self.is_wrapping)  # (1,1,2H,2W)
        residual = self(x)
        return (base + residual).squeeze(0).squeeze(0).clamp(0.0, 1.0)


# ── shared image utilities (used by both train.py and infer.py) ───────────────

def make_gaussian_kernel(size: int, sigma: float,
                         channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    k2d = g.outer(g)
    return k2d.view(1, 1, size, size).expand(channels, 1, size, size).contiguous()


def gaussian_blur(t: torch.Tensor, size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian blur on a (C, H, W) CPU float tensor."""
    C = t.shape[0]
    kernel = make_gaussian_kernel(size, sigma, C, t.device)
    return F.conv2d(t.unsqueeze(0), kernel,
                    padding=size // 2, groups=C).squeeze(0)


def manual_downscale2x(t: torch.Tensor) -> torch.Tensor:
    """
    2× area-average downscale.  No PIL, no extra blur – just averages each 2×2 block.
    t: (C, H, W) where H and W are even.
    """
    C, H, W = t.shape
    return t.reshape(C, H // 2, 2, W // 2, 2).mean(dim=(2, 4))

import numpy as np

def upsample2x(t: torch.Tensor, is_wrapping: bool = False) -> torch.Tensor:
    batched = t.dim() == 4
    if not batched:
        t = t.unsqueeze(0)
    if is_wrapping:
        t_pad = F.pad(t, (1, 1, 1, 1), mode="circular")
    else:
        t_pad = F.pad(t, (1, 1, 1, 1), mode="replicate")

    is_directml = False
    try:
        import torch_directml
        is_directml = True
    except:
        pass
    
    if is_directml:
        # We can't use bilinear mode because the correct corner-centering mode doesn't work
        #  properly on directML devices. Instead we do nearest neighbor than blur. For 2x this
        #  is identical to properly centered bilinear.
        # A shader implementation should just use normal bilinear filtering.
        out_pad = F.interpolate(t_pad, scale_factor=2, mode="nearest")
        
        # We use a 5x5 kernel to implicitly crop off the unused 1->2 pixels of padding on each side.
        kernel = torch.zeros((t.shape[1], 1, 5, 5), device=t.device, dtype=t.dtype)
        
        kernel[:, :, 1, 1] = 1.0/16.0
        kernel[:, :, 2, 1] = 1.0/8.0
        kernel[:, :, 3, 1] = 1.0/16.0
        
        kernel[:, :, 1, 2] = 1.0/8.0
        kernel[:, :, 2, 2] = 1.0/4.0
        kernel[:, :, 3, 2] = 1.0/8.0
        
        kernel[:, :, 1, 3] = 1.0/16.0
        kernel[:, :, 2, 3] = 1.0/8.0
        kernel[:, :, 3, 3] = 1.0/16.0
        
        out = F.conv2d(out_pad, kernel, groups=t.shape[1])
    else:
        # Upstream pytorch. Bilinear with align_corners=True is safe.
        out_pad = F.interpolate(t_pad, scale_factor=2, mode="bilinear", align_corners=False)
        out = out_pad[:,:,2:-2,2:-2]
        
    return out if batched else out.squeeze(0)


def upscale_edi_2x(t: torch.Tensor, is_wrapping: bool = False) -> torch.Tensor:
    """
    Upscales a tensor (C, W, H) or (G, D, W, H) by 2x spatially.
    Follows OpenGL pixel-center logic and supports clamp vs wrap.
    """
    
    device = t.device
    original_dim = t.dim()

    if original_dim == 3:
        t = t.unsqueeze(0)

    g, d, w, h = t.shape
    ow, oh = w * 2, h * 2

    # Pad upfront so all 2x2 neighborhood accesses stay in-bounds.
    # idx_x/idx_y reach -1 at the low end and w/h at the high end,
    # so a 1-pixel border is exactly enough.
    pad_mode = "circular" if is_wrapping else "replicate"
    t_pad = F.pad(t, (1, 1, 1, 1), mode=pad_mode)  # (g, d, w+2, h+2)

    # 1. Coordinate Generation (OpenGL Pixel Center Logic)
    x_coords = (torch.arange(ow, device=device) + 0.5) / 2.0
    y_coords = (torch.arange(oh, device=device) + 0.5) / 2.0
    uvx_x, uvx_y = torch.meshgrid(x_coords, y_coords, indexing='ij')

    # 2. Quad Sampling Indices
    # +1 offset accounts for the 1-pixel pad, so raw idx -1 maps to index 0.
    idx_x = torch.floor(uvx_x - 0.5).long() + 1
    idx_y = torch.floor(uvx_y - 0.5).long() + 1

    # Sample 2x2 neighbourhood — no clamping or wrapping needed here.
    s_a = t_pad[:, :, idx_x,     idx_y    ]  # Top-Left
    s_b = t_pad[:, :, idx_x + 1, idx_y    ]  # Top-Right
    s_c = t_pad[:, :, idx_x,     idx_y + 1]  # Bottom-Left
    s_d = t_pad[:, :, idx_x + 1, idx_y + 1]  # Bottom-Right

    # 3. Local Fractional Coordinates
    tx = (uvx_x - (idx_x - 1 + 0.5)).view(1, 1, ow, oh)  # undo the +1 offset
    ty = (uvx_y - (idx_y - 1 + 0.5)).view(1, 1, ow, oh)

    # 4. Edge Direction Detection
    diff_q = torch.abs(s_d - s_a)
    diff_r = torch.abs(s_c - s_b)
    # dd: 0.0 = edge along A-D diagonal, 1.0 = edge along B-C diagonal
    dd = torch.clamp((diff_r - diff_q) * 8.0, -1.0, 1.0) * 0.5 + 0.5

    # 5. Interpolation iBS (Barycentric Subset)
    mask_bs = (tx + ty < 1.0)
    iBS_1 = s_c * ty + s_b * tx + s_a * (1.0 - (tx + ty))
    iBS_2 = s_c * (1.0-tx) + s_b * (1.0-ty) + s_d * (1.0 - ((1.0-tx) + (1.0-ty)))
    iBS = torch.where(mask_bs, iBS_1, iBS_2)

    # 6. Interpolation iFS (Flipped Subset)
    mask_fs = (tx > ty)
    iFS_1 = s_a * (1.0-tx) + s_d * ty       + s_b * (1.0 - ((1.0-tx) + ty))
    iFS_2 = s_a * (1.0-ty) + s_d * tx       + s_c * (1.0 - (tx + (1.0-ty)))
    iFS = torch.where(mask_fs, iFS_1, iFS_2)

    # 7. Final Mixing
    bilinear = (s_a * (1-tx)*(1-ty)
              + s_b *    tx *(1-ty)
              + s_c * (1-tx)*   ty
              + s_d *    tx *   ty)

    edi_ver = torch.where(torch.round(dd) > 0.5, iFS, iBS)
    mix_factor = torch.abs(dd - 0.5) * 2.0
    res = (1.0 - mix_factor) * bilinear + mix_factor * edi_ver
    
    # 8. high frequency layer
    kc = 3          # kernel half-size
    kg = 2          # tap offset
    ks = 0.5        # strength
    
    bilinear = F.pad(bilinear, (kc, kc, kc, kc), mode=pad_mode)
    kernel = torch.zeros((d, 1, kc*2+1, kc*2+1), device=device, dtype=t.dtype)
    kernel[:, :, kc,    kc-kg] = -ks / 6.0
    kernel[:, :, kc-kg, kc   ] = -ks / 6.0
    kernel[:, :, kc,    kc   ] = -ks / 3.0 + ks
    kernel[:, :, kc+kg, kc   ] = -ks / 6.0
    kernel[:, :, kc,    kc+kg] = -ks / 6.0
    bilinear = F.conv2d(bilinear, kernel, groups=d)
    
    res += bilinear

    return res.squeeze(0) if original_dim == 3 else res
