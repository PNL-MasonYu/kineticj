pro kj_stix_current, overPlotAR2 = _overPlotAR2

if keyword_set(_overPlotAR2) then overPlotAR2 = 1 else overPlotAR2 = 0

@constants

; Get the plasma parameters

arP = ar2_read_runData('./',1)

f = arP.freq
w = 2 * !pi * f
B = sqrt( arP.br^2 + arP.bt^2 + arP.bz^2 )
r = arP.r
nPhi = arP.nPhi
nX = n_elements(arP.br)
density = arP.densitySpec

ar2 = ar2_read_ar2input('./')

amu = ar2.amu
atomicZ = ar2.atomicZ

nS = n_elements(amu)

; Get the E field

arS = ar2_read_solution('./',1)
 
; For each species

kx = 0
T_eV = 2e3
kPar = nPhi / r 
harmonicNumber = 3

windowWidth = 100

jr = complexArr(nX,nS)
jt = complexArr(nX,nS)
jz = complexArr(nX,nS)

run = 1

if run then begin

for s=0,nS-1 do begin
for i=windowWidth/2,nX-windowWidth/2-1 do begin

    print, 'Spec: ', s
    print, 'iX: ', i

    iL = i-windowWidth/2
    iR = i+windowWidth/2 

    ; Extract and window the E field
    
    N = n_elements(arS.E_r[iL:iR])

    Er = arS.E_r[iL:iR] * hanning(N)
    Et = arS.E_t[iL:iR] * hanning(N)
    Ez = arS.E_z[iL:iR] * hanning(N)
    
    ; Forward FFT
    
    Ekr = fft(Er,/center)
    Ekt = fft(Et,/center)
    Ekz = fft(Ez,/center)
    
    dX = r[1]-r[0]
    kxAxis = fIndGen(N) / ( dX * N ) * 2*!Pi 
    dk = kxAxis[1]-kxAxis[0]
    kxAxis = kxAxis - kxAxis[-1]/2 - dk/2
    
    xRange = [-1,1]*600
    
    ;p=plot(kxAxis,abs(Ekr)^2,layout=[1,3,1],xRange=xRange)
    ;p=plot(kxAxis,abs(Ekt)^2,layout=[1,3,2],/current,xRange=xRange)
    ;p=plot(kxAxis,abs(Ekz)^2,layout=[1,3,3],/current,xRange=xRange)

    ; Get sigma for each k 
  
    jkr = complexArr(N)
    jkt = complexArr(N)
    jkz = complexArr(N)

    ;for k=0,N-1 do begin

        epsilon = kj_hot_epsilon( f, amu[s], atomicZ[s], B[i], $
                density[i,0,s], harmonicNumber, kPar[i], kxAxis, T_eV, $
                kx = kx );
            ;epsilon_cold = epsilon_cold, $
            ;epsilon_swan_WD = epsilon_swan, $
            ;epsilon_swan_ND = epsilon_swan_ND, $
            ;kx = kx ) 

        sigma =  ( epsilon - rebin(identity(3),3,3,N) ) * w * _e0 / _ii

        ; Calculate k-space plasma current
        ; This would have to be generalize for non magnetically aligned coordinates.

        ; Try flipping the indexing order also.

        jkr = (sigma[0,0,*] * Ekr + sigma[1,0,*] * Ekz + sigma[2,0,*] * Ekt)[*]
        jkz = (sigma[0,1,*] * Ekr + sigma[1,1,*] * Ekz + sigma[2,1,*] * Ekt)[*]
        jkt = (sigma[0,2,*] * Ekr + sigma[1,2,*] * Ekz + sigma[2,2,*] * Ekt)[*]

    ;endfor
    
    ; Inverse FFT for configuration-space plasma current
    
    thisjr = fft(jkr,/center,/inverse)
    thisjt = fft(jkt,/center,/inverse)
    thisjz = fft(jkz,/center,/inverse)

    jr[i,s] = thisjr[N/2]
    jt[i,s] = thisjt[N/2]
    jz[i,s] = thisjz[N/2]

endfor
endfor

    save, jr, jt, jz, fileName='kj_sc.sav', /variables

endif else begin

    restore, 'kj_sc.sav'

endelse

Er = arS.E_r
Et = arS.E_t
Ez = arS.E_z

p=plot(r,Er,layout=[1,3,1])
p=plot(r,imaginary(Er),color='r',/over)

p=plot(r,Et,layout=[1,3,2],/current)
p=plot(r,imaginary(Et),color='r',/over)

p=plot(r,Ez,layout=[1,3,3],/current)
p=plot(r,imaginary(Ez),color='r',/over)

current=0
for s=0,nS-1 do begin

    p=plot(r,jr[*,s],layout=[nS,3,1+s],current=current)
    p=plot(r,imaginary(jr[*,s]),color='r',/over)
    if overPlotAR2 then begin
        p=plot(arS.r,arS.JP_r[*,0,s],thick=4,transparency=80,/over)            
        p=plot(arS.r,imaginary(arS.JP_r[*,0,s]),color='r',thick=4,transparency=80,/over)            
    endif

    current = (current + 1)<1
    p=plot(r,jt[*,s],layout=[nS,3,1+1*nS+s],current=current)
    p=plot(r,imaginary(jt[*,s]),color='r',/over)
    if overPlotAR2 then begin
        p=plot(arS.r,arS.JP_t[*,0,s],thick=4,transparency=80,/over)            
        p=plot(arS.r,imaginary(arS.JP_t[*,0,s]),color='r',thick=4,transparency=80,/over)            
    endif

    p=plot(r,jz[*,s],layout=[nS,3,1+2*nS+s],current=current)
    p=plot(r,imaginary(jz[*,s]),color='r',/over)
    if overPlotAR2 then begin
        p=plot(arS.r,arS.JP_z[*,0,s],thick=4,transparency=80,/over)            
        p=plot(arS.r,imaginary(arS.JP_z[*,0,s]),color='r',thick=4,transparency=80,/over)            
    endif


endfor

stop
end