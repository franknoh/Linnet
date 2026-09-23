if exists("b:current_syntax")
  finish
endif

syn keyword linnetConditional if else match
syn keyword linnetStatement return where
syn keyword linnetRepeat static for in
syn keyword linnetStructure module use const type struct enum fn op block entry param buffer state sub let var
syn keyword linnetModifier pub as
syn keyword linnetSelf crate self super
syn keyword linnetBoolean true false
syn keyword linnetConstant none
syn keyword linnetOption some
syn keyword linnetReserved state extern while async await effect unsafe macro trait impl derive random rng mut ref yield kernel device

syn keyword linnetType bool i8 i16 i32 i64 u8 u16 u32 u64 f16 bf16 f32 f64 Tensor
syn keyword linnetConstraint Dim Shape DType Numeric Integer Float

syn match linnetTypeName "\<[A-Z][A-Za-z0-9_]*\>"
syn match linnetFunction "\<[a-z_][A-Za-z0-9_]*\ze\s*\(<[^<>()]*>\)\?\s*("
syn match linnetBuiltin "\<\(cast\|reshape\|permute\|broadcast_to\|concat\|pad\|iota shl shr\|fill\|gather\|scatter\|exp\|log\|sqrt\|rsqrt\|sin\|cos\|tanh\|abs\|select\|min\|max\)\ze\s*\(<[^<>()]*>\)\?\s*("
syn match linnetReduction "\<\(sum\|prod\|max\|min\|any\|all\)\ze\s*\(<[^<>\[\]]*>\)\?\s*\["
syn match linnetDeclName "\(\<\(fn\|op\|entry\)\s\+\)\@<=[A-Za-z_][A-Za-z0-9_]*"

syn match linnetNumber "\<0x[0-9A-Fa-f]\(_\?[0-9A-Fa-f]\)*\>"
syn match linnetNumber "\<0b[01]\(_\?[01]\)*\>"
syn match linnetNumber "\<[0-9]\(_\?[0-9]\)*\>"
syn match linnetFloat "\<[0-9]\(_\?[0-9]\)*\.[0-9]\(_\?[0-9]\)*\([eE][+-]\?[0-9]\(_\?[0-9]\)*\)\?\>"
syn match linnetFloat "\<[0-9]\(_\?[0-9]\)*[eE][+-]\?[0-9]\(_\?[0-9]\)*\>"

syn match linnetEscape contained "\\\([nrt0\\\"]\|u{[0-9A-Fa-f]\{1,6}}\)"
syn region linnetString start=+"+ skip=+\\.+ end=+"\|$+ contains=linnetEscape

syn match linnetOperator "->\|=>\|==\|!=\|<=\|>=\|&&\|||\|::\|\.\.\."

syn keyword linnetTodo contained TODO FIXME NOTE
syn match linnetComment "//.*$" contains=linnetTodo,@Spell
syn match linnetDocComment "///\(/\)\@!.*$" contains=linnetTodo,@Spell
syn region linnetBlockComment start="/\*" end="\*/" contains=linnetBlockComment,linnetTodo,@Spell

hi def link linnetConditional Conditional
hi def link linnetStatement Statement
hi def link linnetRepeat Repeat
hi def link linnetStructure Structure
hi def link linnetModifier StorageClass
hi def link linnetSelf Identifier
hi def link linnetBoolean Boolean
hi def link linnetConstant Constant
hi def link linnetOption Function
hi def link linnetReserved Error
hi def link linnetType Type
hi def link linnetConstraint Typedef
hi def link linnetTypeName Type
hi def link linnetFunction Function
hi def link linnetBuiltin Function
hi def link linnetReduction Operator
hi def link linnetDeclName Function
hi def link linnetNumber Number
hi def link linnetFloat Float
hi def link linnetEscape SpecialChar
hi def link linnetString String
hi def link linnetOperator Operator
hi def link linnetTodo Todo
hi def link linnetComment Comment
hi def link linnetDocComment SpecialComment
hi def link linnetBlockComment Comment

let b:current_syntax = "linnet"
