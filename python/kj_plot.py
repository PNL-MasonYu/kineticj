# %%
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

ncIn = Dataset('/home/mason/kineticj/WHAM/high_collision_gpu/iter_7/r_10/input/input-data.nc')

e = 1.60217662e-19
q = 1.0*e 
amu = 1.6605402E-27
m = 2 * amu

xIn = ncIn.variables['r'][:]
brIn = ncIn.variables['B0_r'][:]
btIn = ncIn.variables['B0_p'][:]
bzIn = ncIn.variables['B0_z'][:]
erReIn = ncIn.variables['e_r_re'][:]
erImIn = ncIn.variables['e_r_im'][:]
f = ncIn.variables['freq'][0]

B = np.sqrt(np.power(brIn,2)+np.power(btIn,2)+np.power(bzIn,2))
wc = q*B/m
w = 2*np.pi*f
w_wc = w/wc
mod = np.mod(w_wc,1.0)
kPar = 0
T_eV = 5e3
vth = np.sqrt(2*T_eV*e/m)
v_perp = vth/np.sqrt(2)
rho_larmor = m*v_perp/(q*B) * 2e3

nc = Dataset('/home/mason/kineticj/WHAM/high_collision_gpu/iter_7/r_10/output/jP2.nc')

x = nc.variables['x'][:]


jx_re = nc.variables['j1xc_re'][:]
jx_im = nc.variables['j1xc_im'][:]
jy_re = nc.variables['j1yc_re'][:]
jy_im = nc.variables['j1yc_im'][:]
jz_re = nc.variables['j1zc_re'][:]
jz_im = nc.variables['j1zc_im'][:]


jphi_re = np.arctan2(jy_re,jx_re)
jphi_im = np.arctan2(jy_im,jx_im)
jr_re = np.cos(jphi_re)*jx_re + np.sin(jphi_re)*jy_re
jr_im = np.cos(jphi_im)*jx_im + np.sin(jphi_im)*jy_im
jp_re = -np.sin(jphi_re)*jx_re + np.cos(jphi_re)*jy_re
jp_im = -np.sin(jphi_im)*jx_im + np.cos(jphi_im)*jy_im

xRng = [0,2]

fig, (ax0,ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(nrows=7,figsize=(6, 14))

ax0.plot(x,jx_re, label="real")
ax0.plot(x,jx_im, label="imag")
ax0.set_title('Jx')
ax0.set_xlim(xRng)
ax0.legend(loc = "upper right")

ax1.plot(x,jy_re, label="real")
ax1.plot(x,jy_im, label="imag")
ax1.set_title('Jy')
ax1.set_xlim(xRng)
ax1.legend(loc = "upper right")

ax2.plot(x,jz_re, label="real")
ax2.plot(x,jz_im, label="imag")
ax2.set_title('Jz')
ax2.set_xlim(xRng)
ax2.legend(loc = "upper right")

res = w-kPar*vth-1*wc
ax3.semilogy(xIn,1/np.abs(res))
res = w-kPar*vth-2*wc
ax3.semilogy(xIn,1/np.abs(res))
res = w-kPar*vth-3*wc
ax3.semilogy(xIn,1/np.abs(res))
res = w-kPar*vth-4*wc
ax3.semilogy(xIn,1/np.abs(res))
ax3.set_xlim(xRng)
ax3.set_title('1/(|w-vth*kpar-n*wc|) for n=1-4 at freq %.2e'%(f))

ax4.plot(xIn,B, label="B (T)")
ax4.plot(xIn,rho_larmor, label='gyrodiameter (mm)')
ax4.set_xlim(xRng)
ax4.legend(loc="upper right")
ax4.set_title('B (T) and gyrodiameter')

ax5.plot(x,jr_re, label="real")
ax5.plot(x,jr_im, label="imag")
ax5.set_title('Jr')
ax5.set_xlim(xRng)
ax5.legend(loc = "upper right")

ax6.plot(x,jp_re, label="real")
ax6.plot(x,jp_im, label="imag")
ax6.set_title('Jp')
ax6.set_xlim(xRng)
ax6.legend(loc = "upper right")

fig.subplots_adjust(hspace=0.8)
plt.show()


# %%
