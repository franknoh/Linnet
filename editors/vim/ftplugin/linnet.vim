if exists("b:did_ftplugin")
  finish
endif
let b:did_ftplugin = 1

setlocal commentstring=//\ %s
setlocal comments=:///,://,s1:/*,mb:*,ex:*/
setlocal formatoptions-=t formatoptions+=croql
setlocal expandtab shiftwidth=4 softtabstop=4
setlocal suffixesadd=.linnet

" `gq` and `=` style reformatting of the whole buffer through the formatter:
"   :%!linnet fmt -
if executable("linnet")
  setlocal formatprg=linnet\ fmt\ -
endif

let b:undo_ftplugin = "setlocal commentstring< comments< formatoptions< expandtab< shiftwidth< softtabstop< suffixesadd< formatprg<"
