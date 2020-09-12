
REQUIRES Python 3.7+

TODO:
gather probably can be replaced by boolean_mask, anywhere
Generate text file with train samples, to debug

NON SEQUENTIAL:
Score: 62793 of 151689
Ratio: 0.4139588236457489
Total time: 2.959610939025879

SEQUENTIAL:
Score: 68515 of 150192
Ratio: 0.45618275274315545
Total time: 2.914384126663208

(After pre/post processing / 1)
Score: 68951 of 151984
Ratio: 0.45367275502684495
Total time: 4.415704011917114


See https://stackoverflow.com/questions/50166420/how-to-concatenate-2-embedding-layers-with-mask-zero-true-in-keras2-0 for multiple
masked inputs (for future projects)

Requisites:
tensorflow==2.3
