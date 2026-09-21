if exists("b:did_indent")
  finish
endif
let b:did_indent = 1

setlocal indentexpr=GetLinnetIndent()
setlocal indentkeys=0{,0},0),0],!^F,o,O
setlocal nosmartindent

let b:undo_indent = "setlocal indentexpr< indentkeys< smartindent<"

if exists("*GetLinnetIndent")
  finish
endif

" One level per unclosed bracket on the previous code line; a line that starts
" with a closing bracket goes back one level.
function! GetLinnetIndent()
  let prev = prevnonblank(v:lnum - 1)
  if prev == 0
    return 0
  endif
  let prevline = substitute(getline(prev), '//.*$', '', '')
  let indent = indent(prev)
  if prevline =~ '[{(\[]\s*$'
    let indent += shiftwidth()
  endif
  if prevline =~ '^\s*where\s*$'
    let indent += shiftwidth()
  endif
  if getline(v:lnum) =~ '^\s*[})\]]'
    let indent -= shiftwidth()
  endif
  return indent < 0 ? 0 : indent
endfunction
