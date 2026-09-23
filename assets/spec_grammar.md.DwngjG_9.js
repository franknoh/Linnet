import{_ as s,o as a,c as p,a2 as t}from"./chunks/framework.D66Xq0Kv.js";const _=JSON.parse('{"title":"Grammar","description":"","frontmatter":{},"headers":[],"relativePath":"spec/grammar.md","filePath":"spec/grammar.md","lastUpdated":null}'),e={name:"spec/grammar.md"};function o(u,n,l,i,q,r){return a(),p("div",null,[...n[0]||(n[0]=[t(`<h1 id="grammar" tabindex="-1">Grammar <a class="header-anchor" href="#grammar" aria-label="Permalink to &quot;Grammar&quot;">​</a></h1><p>The consolidated EBNF grammar (<code>spec/grammar.ebnf</code>).</p><div class="language-text vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">text</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>(* Linnet .linnet grammar sketch. Semantic restrictions are specified in prose. *)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>source_file        = module_decl, { use_decl }, { item } ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>module_decl        = &quot;module&quot;, module_path ;</span></span>
<span class="line"><span>module_path        = identifier, { &quot;.&quot;, identifier } ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>use_decl           = &quot;use&quot;, import_path, [ &quot;::&quot;, import_selector ] ;</span></span>
<span class="line"><span>import_path        = ( &quot;std&quot; | &quot;crate&quot; | identifier ), { &quot;.&quot;, identifier } ;</span></span>
<span class="line"><span>import_selector    = identifier | &quot;{&quot;, import_name, { &quot;,&quot;, import_name }, [ &quot;,&quot; ], &quot;}&quot; ;</span></span>
<span class="line"><span>import_name        = identifier, [ &quot;as&quot;, identifier ] ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>item               = [ &quot;pub&quot; ], ( const_decl</span></span>
<span class="line"><span>                                  | type_alias</span></span>
<span class="line"><span>                                  | struct_decl</span></span>
<span class="line"><span>                                  | enum_decl</span></span>
<span class="line"><span>                                  | fn_decl</span></span>
<span class="line"><span>                                  | op_decl</span></span>
<span class="line"><span>                                  | block_decl</span></span>
<span class="line"><span>                                  | entry_decl ) ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>const_decl         = &quot;const&quot;, identifier, [ &quot;:&quot;, type ], &quot;=&quot;, expr ;</span></span>
<span class="line"><span>type_alias         = &quot;type&quot;, identifier, [ generic_params ], &quot;=&quot;, type ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>struct_decl        = &quot;struct&quot;, identifier, [ generic_params ], &quot;{&quot;, { field_decl }, &quot;}&quot; ;</span></span>
<span class="line"><span>field_decl         = identifier, &quot;:&quot;, type ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>enum_decl          = &quot;enum&quot;, identifier, [ generic_params ], &quot;{&quot;, [ identifier, { &quot;,&quot;, identifier }, [ &quot;,&quot; ] ], &quot;}&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>fn_decl            = &quot;fn&quot;, identifier, [ generic_params ], parameter_list,</span></span>
<span class="line"><span>                     [ &quot;-&gt;&quot;, type ], [ where_clause ], block_body ;</span></span>
<span class="line"><span>op_decl            = &quot;op&quot;, identifier, [ generic_params ], parameter_list,</span></span>
<span class="line"><span>                     &quot;-&gt;&quot;, type, [ where_clause ], block_body ;</span></span>
<span class="line"><span>entry_decl         = &quot;entry&quot;, identifier, [ generic_params ], parameter_list,</span></span>
<span class="line"><span>                     [ &quot;-&gt;&quot;, type ], [ where_clause ], block_body ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>block_decl         = &quot;block&quot;, identifier, [ generic_params ], [ where_clause ],</span></span>
<span class="line"><span>                     &quot;{&quot;, { block_item }, &quot;}&quot; ;</span></span>
<span class="line"><span>block_item         = param_decl | buffer_decl | state_decl | sub_decl | item ;</span></span>
<span class="line"><span>param_decl         = &quot;param&quot;, identifier, &quot;:&quot;, type, [ &quot;=&quot;, expr ] ;</span></span>
<span class="line"><span>buffer_decl        = &quot;buffer&quot;, identifier, &quot;:&quot;, type ;</span></span>
<span class="line"><span>state_decl         = &quot;state&quot;, identifier, &quot;:&quot;, type ;</span></span>
<span class="line"><span>sub_decl           = &quot;sub&quot;, identifier, &quot;:&quot;, type ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>generic_params     = &quot;&lt;&quot;, generic_param, { &quot;,&quot;, generic_param }, [ &quot;,&quot; ], &quot;&gt;&quot; ;</span></span>
<span class="line"><span>generic_param      = [ &quot;*&quot; ], identifier, &quot;:&quot;, generic_constraint, [ &quot;=&quot;, type_or_expr ] ;</span></span>
<span class="line"><span>generic_constraint = &quot;Dim&quot; | &quot;Shape&quot; | &quot;DType&quot; | &quot;Numeric&quot; | &quot;Integer&quot; | &quot;Float&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>parameter_list     = &quot;(&quot;, [ parameter, { &quot;,&quot;, parameter }, [ &quot;,&quot; ] ], &quot;)&quot; ;</span></span>
<span class="line"><span>parameter          = identifier, &quot;:&quot;, type, [ &quot;=&quot;, expr ] ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>where_clause       = &quot;where&quot;, constraint, { &quot;,&quot;, constraint }, [ &quot;,&quot; ] ;</span></span>
<span class="line"><span>constraint         = expr, comparison_op, expr ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>block_body         = &quot;{&quot;, { statement }, &quot;}&quot; ;</span></span>
<span class="line"><span>statement          = let_stmt</span></span>
<span class="line"><span>                   | var_stmt</span></span>
<span class="line"><span>                   | assign_stmt</span></span>
<span class="line"><span>                   | return_stmt</span></span>
<span class="line"><span>                   | static_for_stmt</span></span>
<span class="line"><span>                   | while_stmt ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>let_stmt           = &quot;let&quot;, ( pattern, [ &quot;:&quot;, type ], &quot;=&quot;, expr | comprehension_lhs, &quot;=&quot;, expr ) ;</span></span>
<span class="line"><span>var_stmt           = &quot;var&quot;, identifier, [ &quot;:&quot;, type ], &quot;=&quot;, expr ;</span></span>
<span class="line"><span>assign_stmt        = identifier, &quot;=&quot;, expr ;   (* a \`var\` local or a \`state\` member *)</span></span>
<span class="line"><span>return_stmt        = &quot;return&quot;, [ expr ] ;</span></span>
<span class="line"><span>static_for_stmt    = &quot;static&quot;, &quot;for&quot;, pattern, &quot;in&quot;, expr, [ &quot;..&quot;, expr ], block_body ;</span></span>
<span class="line"><span>while_stmt         = &quot;while&quot;, expr, block_body ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>pattern            = identifier</span></span>
<span class="line"><span>                   | &quot;(&quot;, pattern, { &quot;,&quot;, pattern }, [ &quot;,&quot; ], &quot;)&quot;</span></span>
<span class="line"><span>                   | &quot;some&quot;, &quot;(&quot;, pattern, &quot;)&quot;</span></span>
<span class="line"><span>                   | &quot;none&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>comprehension_lhs  = identifier, &quot;[&quot;, index_output, { &quot;,&quot;, index_output }, &quot;]&quot; ;</span></span>
<span class="line"><span>index_output       = identifier | &quot;*&quot;, identifier ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>expr               = if_expr | match_expr | logical_or ;</span></span>
<span class="line"><span>if_expr            = &quot;if&quot;, expr, block_expr, &quot;else&quot;, block_expr ;</span></span>
<span class="line"><span>block_expr         = &quot;{&quot;, expr, &quot;}&quot; ;</span></span>
<span class="line"><span>match_expr         = &quot;match&quot;, expr, &quot;{&quot;, match_arm, { match_arm }, &quot;}&quot; ;</span></span>
<span class="line"><span>(* An arm pattern never starts with &quot;(&quot;: arms have no separator, and a</span></span>
<span class="line"><span>   parenthesized pattern would read as a call on the previous arm&#39;s value. *)</span></span>
<span class="line"><span>match_arm          = arm_pattern, &quot;=&gt;&quot;, expr ;</span></span>
<span class="line"><span>arm_pattern        = identifier | &quot;some&quot;, &quot;(&quot;, pattern, &quot;)&quot; | &quot;none&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>logical_or         = logical_and, { &quot;||&quot;, logical_and } ;</span></span>
<span class="line"><span>logical_and        = equality, { &quot;&amp;&amp;&quot;, equality } ;</span></span>
<span class="line"><span>equality           = comparison, { ( &quot;==&quot; | &quot;!=&quot; ), comparison } ;</span></span>
<span class="line"><span>comparison         = additive, { ( &quot;&lt;&quot; | &quot;&lt;=&quot; | &quot;&gt;&quot; | &quot;&gt;=&quot; ), additive } ;</span></span>
<span class="line"><span>additive           = multiplicative, { ( &quot;+&quot; | &quot;-&quot; ), multiplicative } ;</span></span>
<span class="line"><span>multiplicative     = unary, { ( &quot;*&quot; | &quot;/&quot; | &quot;%&quot; ), unary } ;</span></span>
<span class="line"><span>unary              = [ &quot;!&quot; | &quot;-&quot; | &quot;+&quot; ], postfix ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>postfix            = primary, { call_suffix | index_suffix | member_suffix } ;</span></span>
<span class="line"><span>call_suffix        = [ generic_arguments ], &quot;(&quot;, [ argument, { &quot;,&quot;, argument }, [ &quot;,&quot; ] ], &quot;)&quot; ;</span></span>
<span class="line"><span>generic_arguments  = &quot;&lt;&quot;, type_or_expr, { &quot;,&quot;, type_or_expr }, [ &quot;,&quot; ], &quot;&gt;&quot; ;</span></span>
<span class="line"><span>argument           = [ identifier, &quot;=&quot; ], expr ;</span></span>
<span class="line"><span>member_suffix      = &quot;.&quot;, identifier ;</span></span>
<span class="line"><span>index_suffix       = &quot;[&quot;, index_component, { &quot;,&quot;, index_component }, &quot;]&quot; ;</span></span>
<span class="line"><span>index_component    = expr | &quot;...&quot; | &quot;*&quot;, identifier | slice_component ;</span></span>
<span class="line"><span>slice_component    = [ expr ], &quot;:&quot;, [ expr ], [ &quot;:&quot;, [ expr ] ] ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>primary            = literal</span></span>
<span class="line"><span>                   | identifier</span></span>
<span class="line"><span>                   | tuple_expr</span></span>
<span class="line"><span>                   | shape_literal</span></span>
<span class="line"><span>                   | reduction_expr</span></span>
<span class="line"><span>                   | &quot;some&quot;, &quot;(&quot;, expr, &quot;)&quot;</span></span>
<span class="line"><span>                   | &quot;none&quot;</span></span>
<span class="line"><span>                   | &quot;(&quot;, expr, &quot;)&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tuple_expr         = &quot;(&quot;, expr, &quot;,&quot;, [ expr, { &quot;,&quot;, expr }, [ &quot;,&quot; ] ], &quot;)&quot; ;</span></span>
<span class="line"><span>shape_literal      = &quot;[&quot;, [ expr, { &quot;,&quot;, expr }, [ &quot;,&quot; ] ], &quot;]&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>reduction_expr     = reduction_kind, [ &quot;&lt;&quot;, scalar_type, &quot;&gt;&quot; ],</span></span>
<span class="line"><span>                     &quot;[&quot;, reduction_index, { &quot;,&quot;, reduction_index }, &quot;]&quot;, expr ;</span></span>
<span class="line"><span>(* Reduction names are contextual: they begin a reduction only when followed</span></span>
<span class="line"><span>   by &quot;[&quot; or by &quot;&lt;&quot;, scalar_type, &quot;&gt;&quot;, &quot;[&quot;. Elsewhere they are identifiers. *)</span></span>
<span class="line"><span>reduction_kind     = &quot;sum&quot; | &quot;prod&quot; | &quot;max&quot; | &quot;min&quot; | &quot;any&quot; | &quot;all&quot; ;</span></span>
<span class="line"><span>reduction_index    = identifier ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>type               = type_primary, [ &quot;?&quot; ] ;</span></span>
<span class="line"><span>type_primary       = scalar_type</span></span>
<span class="line"><span>                   | tensor_type</span></span>
<span class="line"><span>                   | tuple_type</span></span>
<span class="line"><span>                   | array_type</span></span>
<span class="line"><span>                   | named_type ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>scalar_type        = &quot;bool&quot;</span></span>
<span class="line"><span>                   | &quot;i8&quot; | &quot;i16&quot; | &quot;i32&quot; | &quot;i64&quot;</span></span>
<span class="line"><span>                   | &quot;u8&quot; | &quot;u16&quot; | &quot;u32&quot; | &quot;u64&quot;</span></span>
<span class="line"><span>                   | &quot;f16&quot; | &quot;bf16&quot; | &quot;f32&quot; | &quot;f64&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tensor_type        = &quot;Tensor&quot;, &quot;[&quot;, shape_spec, &quot;;&quot;, type_atom, &quot;]&quot; ;</span></span>
<span class="line"><span>shape_spec         = [ shape_element, { &quot;,&quot;, shape_element } ] ;</span></span>
<span class="line"><span>shape_element      = expr | &quot;*&quot;, identifier ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>tuple_type         = &quot;(&quot;, type, &quot;,&quot;, [ type, { &quot;,&quot;, type }, [ &quot;,&quot; ] ], &quot;)&quot; ;</span></span>
<span class="line"><span>array_type         = &quot;[&quot;, type, &quot;;&quot;, expr, &quot;]&quot; ;</span></span>
<span class="line"><span>named_type         = module_path, [ generic_arguments ] ;</span></span>
<span class="line"><span>type_atom          = scalar_type | named_type ;</span></span>
<span class="line"><span>(* A generic argument is a type when it parses as one; otherwise it is an</span></span>
<span class="line"><span>   arithmetic expression. Comparison and logical operators must be</span></span>
<span class="line"><span>   parenthesized inside generic argument lists. *)</span></span>
<span class="line"><span>type_or_expr       = type | additive ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>literal            = integer_literal | float_literal | string_literal | &quot;true&quot; | &quot;false&quot; ;</span></span>
<span class="line"><span>comparison_op      = &quot;==&quot; | &quot;!=&quot; | &quot;&lt;&quot; | &quot;&lt;=&quot; | &quot;&gt;&quot; | &quot;&gt;=&quot; ;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>identifier         = ? ASCII [A-Za-z_][A-Za-z0-9_]* ? ;</span></span>
<span class="line"><span>integer_literal    = ? integer token ? ;</span></span>
<span class="line"><span>float_literal      = ? floating token ? ;</span></span>
<span class="line"><span>string_literal     = ? UTF-8 quoted string token ? ;</span></span></code></pre></div>`,3)])])}const m=s(e,[["render",o]]);export{_ as __pageData,m as default};
