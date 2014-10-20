pro kj_sum_spec_jp, FilesToSum, SumFileName = SumFileName

    if keyword_set(SumFileName) then begin
            OutFileName = SumFileName
    endif else begin
            OutFileName = 'kj_jp.nc'
    endelse

    nS = n_elements(FilesToSum)

    kjTmp = kj_read_kj_jp(FilesToSum[0])

    nR = n_elements(kjTmp.jPr)
    nR_ = n_elements(kjTmp.jPr_)

    jPr_spec  = ComplexArr(nR,nS)
    jPr_spec_ = ComplexArr(nR_,nS)
    jPt_spec  = ComplexArr(nR,nS)
    jPt_spec_ = ComplexArr(nR_,nS)
    jPz_spec  = ComplexArr(nR,nS)
    jPz_spec_ = ComplexArr(nR_,nS)


    for s=0,nS-1 do begin

        kj = kj_read_kj_jp(FilesToSum[s])

        if s eq 0 then begin

            jPSum_r =  kj.jPr
            jPSum_t =  kj.jPt
            jPSum_z =  kj.jPz

            jPSum_r_ = kj.jPr_
            jPSum_t_ = kj.jPt_
            jPSum_z_ = kj.jPz_

        endif else begin

            jPSum_r = jPSum_r + kj.jPr
            jPSum_t = jPSum_t + kj.jPt
            jPSum_z = jPSum_z + kj.jPz

            jPSum_r_ = jPSum_r_ + kj.jPr_
            jPSum_t_ = jPSum_t_ + kj.jPt_
            jPSum_z_ = jPSum_z_ + kj.jPz_

        endelse

        jPr_spec[*,s]  = kj.jPr
        jPr_spec_[*,s] = kj.jPr_
        jPt_spec[*,s]  = kj.jPt
        jPt_spec_[*,s] = kj.jPt_
        jPz_spec[*,s]  = kj.jPz
        jPz_spec_[*,s] = kj.jPz_


        yRange = [-4,4]
        p=plot(kj.r, kj.jPr, yRange=yRange, layout=[nS,3,s+1], /current, title=FilesToSum[s])
        p=plot(kj.r, imaginary(kj.jPr), color='r',/over,yRange=yRange)

        p=plot(kj.r, kj.jPt, yRange=yRange, layout=[nS,3,s+nS+1], /current)
        p=plot(kj.r, imaginary(kj.jPt), color='r',/over,yRange=yRange)

        p=plot(kj.r, kj.jPz, yRange=yRange, layout=[nS,3,s+2*nS+1], /current)
        p=plot(kj.r, imaginary(kj.jPz), color='r',/over,yRange=yRange )

    endfor


	; Write kj_jP in file for next iterate

	nc_id = nCdf_create (OutFileName, /clobber )

	nCdf_control, nc_id, /fill
	
	nr_id = nCdf_dimDef ( nc_id, 'nR', n_elements(kj.r) )
	nrH_id = nCdf_dimDef ( nc_id, 'nR_', n_elements(kj.r_) )
	scalar_id = nCdf_dimDef ( nc_id, 'scalar', 1 )
    nS_id = nCdf_dimDef( nc_id, 'nSpec', nS)

	freq_id = nCdf_varDef ( nc_id, 'freq', scalar_id, /float )
	r_id = nCdf_varDef ( nc_id, 'r', nr_id, /float )
	rH_id = nCdf_varDef ( nc_id, 'r_', nrH_id, /float )

	jP_r_re_id = nCdf_varDef ( nc_id, 'jP_r_re', nr_id, /float )
	jP_r_im_id = nCdf_varDef ( nc_id, 'jP_r_im', nr_id, /float )
	jP_p_re_id = nCdf_varDef ( nc_id, 'jP_p_re', nr_id, /float )
	jP_p_im_id = nCdf_varDef ( nc_id, 'jP_p_im', nr_id, /float )
	jP_z_re_id = nCdf_varDef ( nc_id, 'jP_z_re', nr_id, /float )
	jP_z_im_id = nCdf_varDef ( nc_id, 'jP_z_im', nr_id, /float )

	jP_r_re_id_ = nCdf_varDef ( nc_id, 'jP_r_re_', nrH_id, /float )
	jP_r_im_id_ = nCdf_varDef ( nc_id, 'jP_r_im_', nrH_id, /float )
	jP_p_re_id_ = nCdf_varDef ( nc_id, 'jP_p_re_', nrH_id, /float )
	jP_p_im_id_ = nCdf_varDef ( nc_id, 'jP_p_im_', nrH_id, /float )
	jP_z_re_id_ = nCdf_varDef ( nc_id, 'jP_z_re_', nrH_id, /float )
	jP_z_im_id_ = nCdf_varDef ( nc_id, 'jP_z_im_', nrH_id, /float )

	jP_r_re_spec_id = nCdf_varDef ( nc_id, 'jP_r_re_spec', [nr_id,nS_id], /float )
	jP_r_im_spec_id = nCdf_varDef ( nc_id, 'jP_r_im_spec', [nr_id,nS_id], /float )
	jP_p_re_spec_id = nCdf_varDef ( nc_id, 'jP_p_re_spec', [nr_id,nS_id], /float )
	jP_p_im_spec_id = nCdf_varDef ( nc_id, 'jP_p_im_spec', [nr_id,nS_id], /float )
	jP_z_re_spec_id = nCdf_varDef ( nc_id, 'jP_z_re_spec', [nr_id,nS_id], /float )
	jP_z_im_spec_id = nCdf_varDef ( nc_id, 'jP_z_im_spec', [nr_id,nS_id], /float )

	jP_r_re_spec_id_ = nCdf_varDef ( nc_id, 'jP_r_re_spec_', [nrH_id,nS_id], /float )
	jP_r_im_spec_id_ = nCdf_varDef ( nc_id, 'jP_r_im_spec_', [nrH_id,nS_id], /float )
	jP_p_re_spec_id_ = nCdf_varDef ( nc_id, 'jP_p_re_spec_', [nrH_id,nS_id], /float )
	jP_p_im_spec_id_ = nCdf_varDef ( nc_id, 'jP_p_im_spec_', [nrH_id,nS_id], /float )
	jP_z_re_spec_id_ = nCdf_varDef ( nc_id, 'jP_z_re_spec_', [nrH_id,nS_id], /float )
	jP_z_im_spec_id_ = nCdf_varDef ( nc_id, 'jP_z_im_spec_', [nrH_id,nS_id], /float )


	nCdf_control, nc_id, /enDef

	nCdf_varPut, nc_id, freq_id, kj.freq

	nCdf_varPut, nc_id, r_id, kj.r
	nCdf_varPut, nc_id, rH_id, kj.r_

	nCdf_varPut, nc_id, jP_r_re_id, real_part(jPSum_r)
	nCdf_varPut, nc_id, jP_r_im_id, imaginary(jPSum_r) 
	nCdf_varPut, nc_id, jP_p_re_id, real_part(jPSum_t)
	nCdf_varPut, nc_id, jP_p_im_id, imaginary(jPSum_t) 
	nCdf_varPut, nc_id, jP_z_re_id, real_part(jPSum_z)
	nCdf_varPut, nc_id, jP_z_im_id, imaginary(jPSum_z) 

	nCdf_varPut, nc_id, jP_r_re_id_, real_part(jPSum_r_)
	nCdf_varPut, nc_id, jP_r_im_id_, imaginary(jPSum_r_) 
	nCdf_varPut, nc_id, jP_p_re_id_, real_part(jPSum_t_)
	nCdf_varPut, nc_id, jP_p_im_id_, imaginary(jPSum_t_) 
	nCdf_varPut, nc_id, jP_z_re_id_, real_part(jPSum_z_)
	nCdf_varPut, nc_id, jP_z_im_id_, imaginary(jPSum_z_) 

	nCdf_varPut, nc_id, jP_r_re_spec_id, real_part(jPr_spec)
	nCdf_varPut, nc_id, jP_r_im_spec_id, imaginary(jPr_spec) 
	nCdf_varPut, nc_id, jP_p_re_spec_id, real_part(jPt_spec)
	nCdf_varPut, nc_id, jP_p_im_spec_id, imaginary(jPt_spec) 
	nCdf_varPut, nc_id, jP_z_re_spec_id, real_part(jPz_spec)
	nCdf_varPut, nc_id, jP_z_im_spec_id, imaginary(jPz_spec) 

	nCdf_varPut, nc_id, jP_r_re_spec_id_, real_part(jPr_spec_)
	nCdf_varPut, nc_id, jP_r_im_spec_id_, imaginary(jPr_spec_) 
	nCdf_varPut, nc_id, jP_p_re_spec_id_, real_part(jPt_spec_)
	nCdf_varPut, nc_id, jP_p_im_spec_id_, imaginary(jPt_spec_) 
	nCdf_varPut, nc_id, jP_z_re_spec_id_, real_part(jPz_spec_)
	nCdf_varPut, nc_id, jP_z_im_spec_id_, imaginary(jPz_spec_) 


	nCdf_close, nc_id

end
