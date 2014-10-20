function kj_read_kj_jp, FileName, PerSpecies = PerSpecies

    cdfId = ncdf_open(FileName)

    nCdf_varGet, cdfId, 'freq', freq
    nCdf_varGet, cdfId, 'r', r
    nCdf_varGet, cdfId, 'r_', r_

	nCdf_varGet, cdfId, 'jP_r_re', kj_jpR_re
	nCdf_varGet, cdfId, 'jP_r_im', kj_jpR_im
	nCdf_varGet, cdfId, 'jP_p_re', kj_jpT_re
	nCdf_varGet, cdfId, 'jP_p_im', kj_jpT_im
	nCdf_varGet, cdfId, 'jP_z_re', kj_jpZ_re
	nCdf_varGet, cdfId, 'jP_z_im', kj_jpZ_im

	nCdf_varGet, cdfId, 'jP_r_re_', kj_jpR_re_
	nCdf_varGet, cdfId, 'jP_r_im_', kj_jpR_im_
	nCdf_varGet, cdfId, 'jP_p_re_', kj_jpT_re_
	nCdf_varGet, cdfId, 'jP_p_im_', kj_jpT_im_
	nCdf_varGet, cdfId, 'jP_z_re_', kj_jpZ_re_
	nCdf_varGet, cdfId, 'jP_z_im_', kj_jpZ_im_

    if keyword_set(perSpecies) then begin
	nCdf_varGet, cdfId, 'jP_r_re_spec',  kj_jpR_spec_re
	nCdf_varGet, cdfId, 'jP_r_im_spec',  kj_jpR_spec_im
	nCdf_varGet, cdfId, 'jP_p_re_spec',  kj_jpT_spec_re
	nCdf_varGet, cdfId, 'jP_p_im_spec',  kj_jpT_spec_im
	nCdf_varGet, cdfId, 'jP_z_re_spec',  kj_jpZ_spec_re
	nCdf_varGet, cdfId, 'jP_z_im_spec',  kj_jpZ_spec_im

	nCdf_varGet, cdfId, 'jP_r_re_spec_', kj_jpR_spec_re_
	nCdf_varGet, cdfId, 'jP_r_im_spec_', kj_jpR_spec_im_
	nCdf_varGet, cdfId, 'jP_p_re_spec_', kj_jpT_spec_re_
	nCdf_varGet, cdfId, 'jP_p_im_spec_', kj_jpT_spec_im_
	nCdf_varGet, cdfId, 'jP_z_re_spec_', kj_jpZ_spec_re_
	nCdf_varGet, cdfId, 'jP_z_im_spec_', kj_jpZ_spec_im_
    endif

	;nCdf_varGet, cdfId, 'E_r_re', kj_ER_re
	;nCdf_varGet, cdfId, 'E_r_im', kj_ER_im
	;nCdf_varGet, cdfId, 'E_p_re', kj_ET_re
	;nCdf_varGet, cdfId, 'E_p_im', kj_ET_im
	;nCdf_varGet, cdfId, 'E_z_re', kj_EZ_re
	;nCdf_varGet, cdfId, 'E_z_im', kj_EZ_im

	;nCdf_varGet, cdfId, 'E_r_re_', kj_ER_re_
	;nCdf_varGet, cdfId, 'E_r_im_', kj_ER_im_
	;nCdf_varGet, cdfId, 'E_p_re_', kj_ET_re_
	;nCdf_varGet, cdfId, 'E_p_im_', kj_ET_im_
	;nCdf_varGet, cdfId, 'E_z_re_', kj_EZ_re_
	;nCdf_varGet, cdfId, 'E_z_im_', kj_EZ_im_

	ncdf_close, cdfId

	kj_jpR = complex(kj_jpR_re,kj_jpR_im)
	kj_jpT = complex(kj_jpT_re,kj_jpT_im)
	kj_jpZ = complex(kj_jpZ_re,kj_jpZ_im)

	kj_jpR_ = complex(kj_jpR_re_,kj_jpR_im_)
	kj_jpT_ = complex(kj_jpT_re_,kj_jpT_im_)
	kj_jpZ_ = complex(kj_jpZ_re_,kj_jpZ_im_)

    if keyword_set(PerSpecies) then begin
	kj_jpR_spec =  complex(kj_jpR_spec_re, kj_jpR_spec_im)
	kj_jpT_spec =  complex(kj_jpT_spec_re, kj_jpT_spec_im)
	kj_jpZ_spec =  complex(kj_jpZ_spec_re, kj_jpZ_spec_im)

	kj_jpR_spec_ = complex(kj_jpR_spec_re_,kj_jpR_spec_im_)
	kj_jpT_spec_ = complex(kj_jpT_spec_re_,kj_jpT_spec_im_)
	kj_jpZ_spec_ = complex(kj_jpZ_spec_re_,kj_jpZ_spec_im_)
    endif

	;kj_ER = complex(kj_ER_re,kj_ER_im)
	;kj_ET = complex(kj_ET_re,kj_ET_im)
	;kj_EZ = complex(kj_EZ_re,kj_EZ_im)

	;kj_ER_ = complex(kj_ER_re_,kj_ER_im_)
	;kj_ET_ = complex(kj_ET_re_,kj_ET_im_)
	;kj_EZ_ = complex(kj_EZ_re_,kj_EZ_im_)

    if keyword_set(PerSpecies) then begin
    kjIn = { $
            freq : freq, $
            r : r, $
            r_ : r_, $
        jpR : kj_jpR, $
        jpT : kj_jpT, $
        jpZ : kj_jpZ, $
        jpR_ : kj_jpR_, $
        jpT_ : kj_jpT_, $
        jpZ_ : kj_jpZ_, $
        jpR_spec : kj_jpR_spec, $
        jpT_spec : kj_jpT_spec, $
        jpZ_spec : kj_jpZ_spec, $
        jpR_spec_ : kj_jpR_spec_, $
        jpT_spec_ : kj_jpT_spec_, $
        jpZ_spec_ : kj_jpZ_spec_ $
        ;ER : kj_ER, $
        ;ET : kj_ET, $
        ;EZ : kj_EZ, $
        ;ER_ : kj_ER_, $
        ;ET_ : kj_ET_, $
        ;EZ_ : kj_EZ_ $
    }
    endif else begin
    kjIn = { $
            freq : freq, $
            r : r, $
            r_ : r_, $
        jpR : kj_jpR, $
        jpT : kj_jpT, $
        jpZ : kj_jpZ, $
        jpR_ : kj_jpR_, $
        jpT_ : kj_jpT_, $
        jpZ_ : kj_jpZ_ $
        ;ER : kj_ER, $
        ;ET : kj_ET, $
        ;EZ : kj_EZ, $
        ;ER_ : kj_ER_, $
        ;ET_ : kj_ET_, $
        ;EZ_ : kj_EZ_ $
    }
    endelse

    return, kjIn

end
