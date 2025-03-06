response = aia_get_response(/{{ flags | join(',/') }})
; Pull needed elements out of structure
logte = response.logte
resp94 = response.a94.tresp
resp131 = response.a131.tresp
resp171 = response.a171.tresp
resp193 = response.a193.tresp
resp211 = response.a211.tresp
resp335 = response.a335.tresp
; Interpolate
interp_logte = {{ temperature | to_unit('K') | log10 | list }}
interp_resp94 = interpol(resp94,logte,interp_logte)
interp_resp131 = interpol(resp131,logte,interp_logte)
interp_resp171 = interpol(resp171,logte,interp_logte)
interp_resp193 = interpol(resp193,logte,interp_logte)
interp_resp211 = interpol(resp211,logte,interp_logte)
interp_resp335 = interpol(resp335,logte,interp_logte)