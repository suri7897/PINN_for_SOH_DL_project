PINN4SOH - Plotter
---
>[!NOTE]
> These files provide code for generating the figures, implemented based on the original code from the paper.  
> The codes have been slightly modified to fit the NASA dataset.  
> In particular, the code for Figure 3 was implemented by myself, since the original implementation for this figure was not provided.


Overview
---
The `Plotter` folder contains the codes for generating the **figures** presented in the original paper.  
You can execute these python files to save each `figures`.

```bash
# current directory : PINN4SOH_NASA_dataset/plotter
python3 [filename].py
```
In particular, `Figure 4b.py` can plot the violin plot even **without the CNN results**.
```
python3 Figure\ 4b.py --CNN=False
```