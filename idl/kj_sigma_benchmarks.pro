pro kj_sigma_benchmarks, runKJ=runKJ, $
        benchmark = _benchmark

if keyword_set(_benchmark) then benchmark = _benchmark else benchmark = 1

@constants

n = 100
n_kj = 10

if benchmark eq 1 then begin

    ; Benchmark 1
    ; -----------
    ; Te Scan, D, ky = 0

    f = 13d6
    Z = 1d0
    amu =  2.0;_me_amu
    BUnit = [0,0,1]
    density = 2d19
    harmonicNumber = 4
  
    B_T = [1d0]
    B_T_kj = B_T

    ; Analytic calculation range

    tMin = 0.1
    tMax = 10e3
    T_eV = 10d0^(findGen(n)/(n-1)*(alog10(tMax)-alog10(tMin))+alog10(tMin)) 

    ; KJ calculation range

    tMin_kj = 100 
    tMax_kj = 10e3
    T_eV_kj = 10d0^(findGen(n_kj)/(n_kj-1)*(alog10(tMax_kj)-alog10(tMin_kj))+alog10(tMin_kj)) 
   
    kx = 10.0
    ky = 0.0 
    kz = 100.0

    ; KJ config parameters

    kj_nPx = 21
    kj_nPy = 21
    kj_nPz = 65
    kj_nStepsPerCycle = 100.0
    kj_nRFCycles = 10.0 

endif else if benchmark eq 2 then begin

    ; Benchmark 2
    ; -----------
    ; Te Scan, D, ky != 0

    f = 13d6
    Z = 1d0
    amu =  2.0;_me_amu
    BUnit = [0,0,1]
    density = 2d19
    harmonicNumber = 4

    B_T = [1d0]
    B_T_kj = B_T
   
    ; Analytic calculation range

    tMin = 0.1
    tMax = 10e3
    T_eV = 10d0^(findGen(n)/(n-1)*(alog10(tMax)-alog10(tMin))+alog10(tMin)) 
   
    ; KJ calculation range

    tMin_kj = 100 
    tMax_kj = 10e3
    T_eV_kj = 10d0^(findGen(n_kj)/(n_kj-1)*(alog10(tMax_kj)-alog10(tMin_kj))+alog10(tMin_kj)) 
 
    kx = 10.0
    ky = 23.0 
    kz = 100.0

    ; KJ config parameters

    kj_nPx = 21
    kj_nPy = 21
    kj_nPz = 65
    kj_nStepsPerCycle = 100.0
    kj_nRFCycles = 10.0 

endif else if benchmark eq 3 then begin

    ; Benchmark 3
    ; -----------
    ; B Scan over a few ion-cyclotron resonances
    
    f = 28d9
    Z = 1d0
    amu =  _me_amu
    B = 1d0
    BUnit = [0,0,1]
    density = 5d19
    harmonicNumber = 6
    
    T_eV = [1e3] 
    T_eV_kj = T_eV

    ; Analytic calculation range

    bMin = 0.8
    bMax = 1.5
    B_T = fIndGen(n)/(n-1)*(bMax-bMin)+bMin 

    ; KJ calculation range

    bMin_kj = 1.5
    bMax_kj = 3.0
    B_T_kj = fIndGen(n_kj)/(n_kj-1)*(bMax_kj-bMin_kj)+bMin_kj 

    kx = 10.0
    ky = 20.0 
    kz = 100.0

    ; KJ config parameters

    kj_nPx = 61
    kj_nPy = 61
    kj_nPz = 65
    kj_nStepsPerCycle = 100.0
    kj_nRFCycles = 10.0 

    ; Diagnose the scenario

    beta_ = density * _kB * T_eV / ( B_T^2 / ( 2 *_u0 ) )
    vTh = sqrt( 2.0 * T_eV * _e / ( amu * _amu ) )
    wc = ( Z * _e ) * B_T / ( amu * _amu )
    wp = sqrt ( density * _e^2 / ( amu * _amu * _e0 ) )
 
stop
endif

nT = n_elements(T_eV)
nB = n_elements(B_T)

nT_kj = n_elements(T_eV_kj)
nB_kj = n_elements(B_T_kj)

kPar = kz  
kPer = sqrt( kx^2 + ky^2 ) 

eps = ComplexArr(3,3,n)
eps_swan = ComplexArr(3,3,n)
eps_cold = ComplexArr(3,3,n)

sig = ComplexArr(3,3,n)
sig_swan = ComplexArr(3,3,n)
sig_cold = ComplexArr(3,3,n)

w = 2*!pi*f
cnt = 0
for i=0,nT-1 do begin
for j=0,nB-1 do begin

    thisTeV = T_eV[i]
    thisB = B_T[j]
    this_eps = kj_hot_epsilon(f, amu, Z, thisB, density, $
            harmonicNumber, kPar, kPer, thisTeV, $
            epsilon_cold = this_eps_cold, kx = kx, $
            epsilon_swan_WD = this_eps_swan, $
            epsilon_swan_ND = this_eps_swan_ND  )
    
    eps[*,*,cnt] = this_eps 
    eps_swan[*,*,cnt] = this_eps_swan 
    eps_cold[*,*,cnt] = this_eps_cold 

    sig[*,*,cnt] = (eps[*,*,cnt] - identity(3)) * w * _e0 / _ii
    sig_swan[*,*,cnt] = (eps_swan[*,*,cnt] - identity(3)) * w * _e0 / _ii
    sig_cold[*,*,cnt] = (eps_cold[*,*,cnt] - identity(3)) * w * _e0 / _ii
    ++cnt

endfor
endfor

; Stage and run kj over this range of temperatures 


sig_kj = ComplexArr(3,3,n_kj)

TemplateRunDir = 'benchmark-perp'
RootDir = expand_path('./')
cd, RootDir

cnt = 0
for t=0,nT_kj-1 do begin
for b=0,nB_kj-1 do begin
    
    ThisRunDir = string(t,format='(i5.5)')

    if keyword_set(runKJ) then begin

        file_delete, ThisRunDir, /recursive, /allow_nonexistent

        file_copy, TemplateRunDir, ThisRunDir, /recursive

    endif

    cd, ThisRunDir

    for row=0,2 do begin

        RowString = string(row,format='(i1.1)')
        This_E_FileName = 'input/kj_single_k_' + RowString
        This_jP2_FileName = 'jP2_' + RowString + '.nc'

        if row eq 0 then begin
            E1=1
            E2=0
            E3=0
        endif

        if row eq 1 then begin
            E1=0
            E2=1
            E3=0
        endif

        if row eq 2 then begin
            E1=0
            E2=0
            E3=1
        endif

        kj_create_single_k_input, b0=B_T_kj[b], bUnit=bUnit, kx=kx, f_Hz=f, n_m3=density, $
                Er=Er, Et=Et, Ez=Ez, x=x, writeOutput=runKJ, $
                E1Multiplier=E1, E2Multiplier=E2, E3Multiplier=E3, fileName=This_E_FileName

        if keyword_set(runKJ) then begin

            ; Stage the input wave fields

            ; Adjust the kj.cfg config file parameters

            kj = kj_read_cfg('./')
            kj['eField_fName'] = this_E_FileName
            kj['xGridMin'] = x[0]+(x[-1]-x[0])/2 - (x[1]-x[0])
            kj['xGridMax'] = x[0]+(x[-1]-x[0])/2 + (x[1]-x[0])
            kj['T_keV'] = T_eV_kj[t]*1e-3 
            kj['species_amu'] = float(amu)
            kj['species_Z'] = float(Z)
            kj['ky'] = float(ky)
            kj['kz'] = float(kz) 
            kj['nStepsPerCycle'] = float(kj_nStepsPerCycle) 
            kj['nRFCycles'] = float(kj_nRFCycles)
            kj['nPx'] = fix(kj_nPx) 
            kj['nPy'] = fix(kj_nPy) 
            kj['nPz'] = fix(kj_nPz) 

            ; Set dt (kj_nStepsPerCycle) such that we sample the shortest wavelength
            ; at the highest velocity with adequote sampling

            nvTh = 3
            vThMax = nvTh * sqrt( 2.0 * T_eV_kj[t] * _e / ( amu * _amu ) )
            parSamples = ( 2 * !pi / kPar ) / ( 1 / f / kj_nStepsPerCycle * vThMax )
            print, 'T_eV: ', T_eV_kj[t], '  parSamples: ', parSamples
            perSamples = ( 2 * !pi / kPer ) / ( 1 / f / kj_nStepsPerCycle * vThMax )
            print, 'T_eV: ', T_eV_kj[t], '  perSamples: ', perSamples

            nMinSamples = 5
            par_dt = ( 2 * !pi / kPar ) / nMinSamples / vThMax 
            nHarmonic = 3
            wc = ( Z * _e ) * B_T_kj[b] / ( amu * _amu )
            per_dt = 1 / ( wc / 2 * !pi ) / ( nHarmonic * 2 )

            par_nStepsPerCycle = 1 / ( wc / 2 * !pi ) / par_dt 
            per_nStepsPerCycle = 1 / ( wc / 2 * !pi ) / per_dt 

            print, 'par_nStepsPerCycle: ', par_nStepsPerCycle
            print, 'per_nStepsPerCycle: ', per_nStepsPerCycle

            vPhsPer = w / kPer
            vPhsPar = w / kPar

            kj_write_kj_cfg, kj, './'

            ; Run kj

            RunCommand = '~/code/kineticj/bin/kineticj'
            spawn, RunCommand, StdOut, StdErr
            print, StdOut
            print, StdErr

            file_move, 'jP2.nc', This_jP2_FileName 

        endif

        ; Read in kj results

        ;kj_read_jp_old, x=kj_x, j1x=kj_j1x, j1y=kj_j1y, j1z=kj_j1z, /oldFormat
        kj_read_jp_old, x=kj_x, j1x=kj_j1x, j1y=kj_j1y, j1z=kj_j1z, fileName=This_jP2_FileName
   
        kj_Er = interpol(Er,x,kj_x)
        kj_Et = interpol(Et,x,kj_x)
        kj_Ez = interpol(Ez,x,kj_x)
   
        if row eq 0 then begin 
            sig_kj[row,0,cnt] = (kj_j1x/kj_Er)[0]
            sig_kj[row,1,cnt] = (kj_j1y/kj_Er)[0]
            sig_kj[row,2,cnt] = (kj_j1z/kj_Er)[0]
        endif

        if row eq 1 then begin 
            sig_kj[row,0,cnt] = (kj_j1x/kj_Et)[0]
            sig_kj[row,1,cnt] = (kj_j1y/kj_Et)[0]
            sig_kj[row,2,cnt] = (kj_j1z/kj_Et)[0]
        endif

        if row eq 2 then begin 
            sig_kj[row,0,cnt] = (kj_j1x/kj_Ez)[0]
            sig_kj[row,1,cnt] = (kj_j1y/kj_Ez)[0]
            sig_kj[row,2,cnt] = (kj_j1z/kj_Ez)[0]
        endif

    endfor

    cd, RootDir

    sig_kj[*,*,cnt] = transpose(sig_kj[*,*,cnt])

    ++cnt

endfor
endfor

; Plot results

layout=[3,3]
pos = 1
thick = 2 
style = '--'
transparency = 50
xFS = 6
yFS = 6
margin = [0.15,0.15,0.1,0.15]

plotThis = sig
plotThis_cold = sig_cold

if benchmark eq 1 or benchmark eq 2 then begin
    xTitle ='log10( T [eV] )'
    x = alog10(T_eV)
    x_kj = alog10(T_eV_kj)
endif

if benchmark eq 3 then begin
    xTitle ='$\omega/\omega_{c}$'

    wc = ( Z * _e ) * B_T / ( amu * _amu )
    wc_kj = ( Z * _e ) * B_T_kj / ( amu * _amu )

    x = w / wc
    x_kj = w / wc_kj
endif

p=plot(x,plotThis[0,0,*],layout=[[layout],pos],$
        title='$\sigma_{xx}$',yRange=[-1,1]*max(abs(plotThis[0,0,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{xx} [S/m]$', margin=margin )
p=plot(x,imaginary(plotThis[0,0,*]),color='r',/over)
p=plot(x,plotThis_cold[0,0,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[0,0,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[0,0,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[0,0,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[0,0,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[0,0,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[0,1,*],layout=[[layout],pos],/current, $
        title='$\sigma_{xy}$',yRange=[-1,1]*max(abs(plotThis[0,1,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{xy} [S/m]$', margin=margin )
p=plot(x,imaginary(plotThis[0,1,*]),color='r',/over)
p=plot(x,plotThis_cold[0,1,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[0,1,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[0,1,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[0,1,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[0,1,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[0,1,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[0,2,*],layout=[[layout],pos],/current, $
        title='$\sigma_{xz}$',yRange=[-1,1]*max(abs(plotThis[0,2,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{xz} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[0,2,*]),color='r',/over)
p=plot(x,plotThis_cold[0,2,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[0,2,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[0,2,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[0,2,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[0,2,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[0,2,*]), color='m', /over, thick=1, lineStyle='--')

++pos
p=plot(x,plotThis[1,0,*],layout=[[layout],pos],/current, $
        title='$\sigma_{yx}$',yRange=[-1,1]*max(abs(plotThis[1,0,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{yx} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[1,0,*]),color='r',/over)
p=plot(x,plotThis_cold[1,0,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[1,0,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[1,0,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[1,0,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[1,0,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[1,0,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[1,1,*],layout=[[layout],pos],/current, $
        title='$\sigma_{yy}$',yRange=[-1,1]*max(abs(plotThis[1,1,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{yy} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[1,1,*]),color='r',/over)
p=plot(x,plotThis_cold[1,1,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[1,1,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[1,1,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[1,1,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[1,1,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[1,1,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[1,2,*],layout=[[layout],pos],/current, $
        title='$\sigma_{yz}$',yRange=[-1,1]*max(abs(plotThis[1,2,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{yz} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[1,2,*]),color='r',/over)
p=plot(x,plotThis_cold[1,2,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[1,2,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[1,2,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[1,2,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[1,2,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[1,2,*]), color='m', /over, thick=1, lineStyle='--')

++pos
p=plot(x,plotThis[2,0,*],layout=[[layout],pos],/current, $
        title='$\sigma_{zx}$',yRange=[-1,1]*max(abs(plotThis[2,0,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{zx} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[2,0,*]),color='r',/over)
p=plot(x,plotThis_cold[2,0,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[2,0,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[2,0,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[2,0,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[2,0,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[2,0,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[2,1,*],layout=[[layout],pos],/current, $
        title='$\sigma_{zy}$',yRange=[-1,1]*max(abs(plotThis[2,1,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{zy} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[2,1,*]),color='r',/over)
p=plot(x,plotThis_cold[2,1,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[2,1,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[2,1,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[2,1,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[2,1,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[2,1,*]), color='m', /over, thick=1, lineStyle='--')

++pos 
p=plot(x,plotThis[2,2,*],layout=[[layout],pos],/current, $
        title='$\sigma_{zz}$',yRange=[-1,1]*max(abs(plotThis[2,2,*])),/buffer,$
        font_size=12, xTitle=xTitle, xTickFont_size=xFS, yTickFont_size=yFS, $
        xMinor = 0, axis_style=1, yTitle='$\sigma_{zz} [S/m]$', margin=margin )

p=plot(x,imaginary(plotThis[2,2,*]),color='r',/over)
p=plot(x,plotThis_cold[2,2,*],/over,thick=thick,transparency=transparency,LineStyle=style)
p=plot(x,imaginary(plotThis_cold[2,2,*]),color='r',/over,thick=thick,transparency=transparency,LineStyle=style)

p=plot(x_kj, sig_kj[2,2,*], /over, thick=2)
p=plot(x_kj, imaginary(sig_kj[2,2,*]), color='r', /over, thick=2)

p=plot(x, sig_swan[2,2,*], /over, thick=1, color='m', lineStyle='--')
p=plot(x, imaginary(sig_swan[2,2,*]), color='m', /over, thick=1, lineStyle='--')

p.save, 'kj_sigma_vs_t.png', resolution=300, /transparent
p.save, 'kj_sigma_vs_t.pdf'

stop

end