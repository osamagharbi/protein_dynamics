# DSR paper experiment map

The paper reports four relevant comparison families:

1. Aβ SDF/reconstruction demonstration (Section 4.1).
2. Training-frame reconstruction across trajectories (Table 1).
3. Temporal interpolation against linear interpolation (Section 4.3.1 and
   appendix tables).
4. Future prediction for the next 10 and next 11--20 frames (Table 2), plus
   a four-trajectory GPCRmd case study.

The authoritative source is the published DSR paper:
<https://www.proceedings.com/content/075/075280-0611open.pdf>

The DSR paper explicitly says that Table 1 evaluates training frames. It uses
the first 1500 frames diluted by ten for AdK equilibrium, NhaA, YiiP, and
DRYAD_MD; every second frame in the first 80 for DIMS; every third frame in
the first 120 for FRODA; and every fifth frame in the first 400 for I-FABP.

The paper reports Aβ values of IoU `0.9361±0.0265`, Chamfer
`0.0009±0.0008`, and NC `0.9967±0.0011` for its Section 4.1 experiment.
