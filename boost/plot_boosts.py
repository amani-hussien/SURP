from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


results_dir = Path("/sims") 

plt.figure(figsize=(8, 6))

dataCOLA = np.load(results_dir / f"boostCOLAz0_p10_dstar4_R10.npz")
dataNbody = np.load(results_dir / f"boostNbodyz0_p10_dstar4_R10.npz")
k0 = dataCOLA["k"]
boost0c = dataCOLA["boostPk"]
boost0n = dataNbody['boostPk']
dataCOLA = np.load(results_dir / f"boostCOLAz1_p10_dstar4_R10.npz")
dataNbody = np.load(results_dir / f"boostNbodyz1_p10_dstar4_R10.npz")
k1 = dataCOLA["k"]
boost1c = dataCOLA["boostPk"]
boost1n = dataNbody['boostPk']
dataCOLA = np.load(results_dir / f"boostCOLAz2_p10_dstar4_R10.npz")
dataNbody = np.load(results_dir / f"boostNbodyz2_p10_dstar4_R10.npz")
k2 = dataCOLA["k"]
boost2c = dataCOLA["boostPk"]
boost2n = dataNbody['boostPk']

plt.semilogx(k0, boost0c, label=f"COLA z=0", color='teal', linestyle='dotted')
plt.semilogx(k0, boost0n, label=f"N-body z=0", color='mediumorchid', linestyle='dotted')
plt.semilogx(k1, boost1c, label=f"COLA z=1",  color='teal', linestyle='dashed')
plt.semilogx(k1, boost1n, label=f"N-body z=1", color='mediumorchid', linestyle='dashed')
plt.semilogx(k2, boost2c, label=f"COLA z=2",  color='teal')
plt.semilogx(k2, boost2n, label=f"N-body z=2", color='mediumorchid')


plt.xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$B_{mm}(z,k)$")
# plt.title(r"$R=15,\;p=10,\;d_\star=4$")
plt.title(r"$B(z,k)$- matter field")
plt.grid(True, which="both", alpha=0.3)
plt.legend(ncol=3, fontsize=8, loc='lower right')
plt.tight_layout()
plt.show()