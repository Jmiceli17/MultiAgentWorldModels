ÚÞ
ä
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8û

 Adam/mu_logstd_logmix_net/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:þ*1
shared_name" Adam/mu_logstd_logmix_net/bias/v

4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/v*
_output_shapes	
:þ*
dtype0
¢
"Adam/mu_logstd_logmix_net/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
þ*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/v

6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/v* 
_output_shapes
:
þ*
dtype0

Adam/rnn/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/rnn/lstm_cell/bias/v

-Adam/rnn/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn/lstm_cell/bias/v*
_output_shapes	
:*
dtype0
¨
%Adam/rnn/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/rnn/lstm_cell/recurrent_kernel/v
¡
9Adam/rnn/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp%Adam/rnn/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

Adam/rnn/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*,
shared_nameAdam/rnn/lstm_cell/kernel/v

/Adam/rnn/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn/lstm_cell/kernel/v*
_output_shapes
:	&*
dtype0

 Adam/mu_logstd_logmix_net/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:þ*1
shared_name" Adam/mu_logstd_logmix_net/bias/m

4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOpReadVariableOp Adam/mu_logstd_logmix_net/bias/m*
_output_shapes	
:þ*
dtype0
¢
"Adam/mu_logstd_logmix_net/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
þ*3
shared_name$"Adam/mu_logstd_logmix_net/kernel/m

6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/mu_logstd_logmix_net/kernel/m* 
_output_shapes
:
þ*
dtype0

Adam/rnn/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/rnn/lstm_cell/bias/m

-Adam/rnn/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn/lstm_cell/bias/m*
_output_shapes	
:*
dtype0
¨
%Adam/rnn/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%Adam/rnn/lstm_cell/recurrent_kernel/m
¡
9Adam/rnn/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp%Adam/rnn/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

Adam/rnn/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*,
shared_nameAdam/rnn/lstm_cell/kernel/m

/Adam/rnn/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn/lstm_cell/kernel/m*
_output_shapes
:	&*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

mu_logstd_logmix_net/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:þ**
shared_namemu_logstd_logmix_net/bias

-mu_logstd_logmix_net/bias/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/bias*
_output_shapes	
:þ*
dtype0

mu_logstd_logmix_net/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
þ*,
shared_namemu_logstd_logmix_net/kernel

/mu_logstd_logmix_net/kernel/Read/ReadVariableOpReadVariableOpmu_logstd_logmix_net/kernel* 
_output_shapes
:
þ*
dtype0
}
rnn/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namernn/lstm_cell/bias
v
&rnn/lstm_cell/bias/Read/ReadVariableOpReadVariableOprnn/lstm_cell/bias*
_output_shapes	
:*
dtype0

rnn/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*/
shared_name rnn/lstm_cell/recurrent_kernel

2rnn/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOprnn/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

rnn/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	&*%
shared_namernn/lstm_cell/kernel
~
(rnn/lstm_cell/kernel/Read/ReadVariableOpReadVariableOprnn/lstm_cell/kernel*
_output_shapes
:	&*
dtype0

serving_default_input_1Placeholder*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&*
dtype0*!
shape:ÿÿÿÿÿÿÿÿÿè&
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1rnn/lstm_cell/kernelrnn/lstm_cell/recurrent_kernelrnn/lstm_cell/biasmu_logstd_logmix_net/kernelmu_logstd_logmix_net/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35311610

NoOpNoOp
Ô/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*/
value/B/ Bû.
í
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	inference_base

out_net

signatures*
'
0
1
2
3
4*
'
0
1
2
3
4*
* 
°
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
trace_0
trace_1
trace_2
trace_3* 
6
trace_0
trace_1
trace_2
trace_3* 
* 

iter

beta_1

 beta_2
	!decay
"learning_ratemqmrmsmtmuvvvwvxvyvz*
ª
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)cell
*
state_spec*
·
+layer_with_weights-0
+layer-0
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses*

2serving_default* 
TN
VARIABLE_VALUErnn/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUErnn/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUErnn/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmu_logstd_logmix_net/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEmu_logstd_logmix_net/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 

	0

1*

30*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 


4states
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
ã
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
I
state_size

kernel
recurrent_kernel
bias*
* 
¦
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias*

0
1*

0
1*
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses*
6
Utrace_0
Vtrace_1
Wtrace_2
Xtrace_3* 
6
Ytrace_0
Ztrace_1
[trace_2
\trace_3* 
* 
8
]	variables
^	keras_api
	_total
	`count*
* 
* 

)0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
2*

0
1
2*
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

ftrace_0
gtrace_1* 

htrace_0
itrace_1* 
* 
* 

0
1*

0
1*
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
* 

+0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

_0
`1*

]	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
wq
VARIABLE_VALUEAdam/rnn/lstm_cell/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/rnn/lstm_cell/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/rnn/lstm_cell/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/mu_logstd_logmix_net/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/mu_logstd_logmix_net/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEAdam/rnn/lstm_cell/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUE%Adam/rnn/lstm_cell/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEAdam/rnn/lstm_cell/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/mu_logstd_logmix_net/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/mu_logstd_logmix_net/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(rnn/lstm_cell/kernel/Read/ReadVariableOp2rnn/lstm_cell/recurrent_kernel/Read/ReadVariableOp&rnn/lstm_cell/bias/Read/ReadVariableOp/mu_logstd_logmix_net/kernel/Read/ReadVariableOp-mu_logstd_logmix_net/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/Adam/rnn/lstm_cell/kernel/m/Read/ReadVariableOp9Adam/rnn/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp-Adam/rnn/lstm_cell/bias/m/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/m/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/m/Read/ReadVariableOp/Adam/rnn/lstm_cell/kernel/v/Read/ReadVariableOp9Adam/rnn/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp-Adam/rnn/lstm_cell/bias/v/Read/ReadVariableOp6Adam/mu_logstd_logmix_net/kernel/v/Read/ReadVariableOp4Adam/mu_logstd_logmix_net/bias/v/Read/ReadVariableOpConst*#
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_35312822
Í
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamernn/lstm_cell/kernelrnn/lstm_cell/recurrent_kernelrnn/lstm_cell/biasmu_logstd_logmix_net/kernelmu_logstd_logmix_net/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/rnn/lstm_cell/kernel/m%Adam/rnn/lstm_cell/recurrent_kernel/mAdam/rnn/lstm_cell/bias/m"Adam/mu_logstd_logmix_net/kernel/m Adam/mu_logstd_logmix_net/bias/mAdam/rnn/lstm_cell/kernel/v%Adam/rnn/lstm_cell/recurrent_kernel/vAdam/rnn/lstm_cell/bias/v"Adam/mu_logstd_logmix_net/kernel/v Adam/mu_logstd_logmix_net/bias/v*"
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_35312898®
ô
ö
,__inference_lstm_cell_layer_call_fn_35312650

inputs
states_0
states_1
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310886p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ÙZ
¼
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311791

inputs?
,rnn_lstm_cell_matmul_readvariableop_resource:	&B
.rnn_lstm_cell_matmul_1_readvariableop_resource:
<
-rnn_lstm_cell_biasadd_readvariableop_resource:	R
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource:
þN
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource:	þ
identity¢$rnn/lstm_cell/BiasAdd/ReadVariableOp¢#rnn/lstm_cell/MatMul/ReadVariableOp¢%rnn/lstm_cell/MatMul_1/ReadVariableOp¢	rnn/while¢6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp¢5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp^
rnn/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&E
	rnn/ShapeShapernn/Cast:y:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn/transpose	Transposernn/Cast:y:0rnn/transpose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ì
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
#rnn/lstm_cell/MatMul/ReadVariableOpReadVariableOp,rnn_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
rnn/lstm_cell/MatMulMatMulrnn/strided_slice_2:output:0+rnn/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp.rnn_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
rnn/lstm_cell/MatMul_1MatMulrnn/zeros:output:0-rnn/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/addAddV2rnn/lstm_cell/MatMul:product:0 rnn/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$rnn/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp-rnn_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
rnn/lstm_cell/BiasAddBiasAddrnn/lstm_cell/add:z:0,rnn/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
rnn/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ä
rnn/lstm_cell/splitSplit&rnn/lstm_cell/split/split_dim:output:0rnn/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitq
rnn/lstm_cell/SigmoidSigmoidrnn/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_1Sigmoidrnn/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rnn/lstm_cell/mulMulrnn/lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
rnn/lstm_cell/TanhTanhrnn/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_1Mulrnn/lstm_cell/Sigmoid:y:0rnn/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/add_1AddV2rnn/lstm_cell/mul:z:0rnn/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_2Sigmoidrnn/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
rnn/lstm_cell/Tanh_1Tanhrnn/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_2Mulrnn/lstm_cell/Sigmoid_2:y:0rnn/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0,rnn_lstm_cell_matmul_readvariableop_resource.rnn_lstm_cell_matmul_1_readvariableop_resource-rnn_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
rnn_while_body_35311700*#
condR
rnn_while_cond_35311699*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   r
ReshapeReshapernn/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0´
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ³
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0×
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity0sequential/mu_logstd_logmix_net/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ¸
NoOpNoOp%^rnn/lstm_cell/BiasAdd/ReadVariableOp$^rnn/lstm_cell/MatMul/ReadVariableOp&^rnn/lstm_cell/MatMul_1/ReadVariableOp
^rnn/while7^sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6^sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2L
$rnn/lstm_cell/BiasAdd/ReadVariableOp$rnn/lstm_cell/BiasAdd/ReadVariableOp2J
#rnn/lstm_cell/MatMul/ReadVariableOp#rnn/lstm_cell/MatMul/ReadVariableOp2N
%rnn/lstm_cell/MatMul_1/ReadVariableOp%rnn/lstm_cell/MatMul_1/ReadVariableOp2
	rnn/while	rnn/while2p
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp2n
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Ë"
Þ
while_body_35310945
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_35310969_0:	&.
while_lstm_cell_35310971_0:
)
while_lstm_cell_35310973_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_35310969:	&,
while_lstm_cell_35310971:
'
while_lstm_cell_35310973:	¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0¯
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35310969_0while_lstm_cell_35310971_0while_lstm_cell_35310973_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310886Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_35310969while_lstm_cell_35310969_0"6
while_lstm_cell_35310971while_lstm_cell_35310971_0"6
while_lstm_cell_35310973while_lstm_cell_35310973_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ä6
¹
while_body_35312493
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ö

G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310735

inputs

states
states_11
matmul_readvariableop_resource:	&4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	&*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
à	

R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045

inputs2
matmul_readvariableop_resource:
þ.
biasadd_readvariableop_resource:	þ
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
Í
while_cond_35311380
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35311380___redundant_placeholder06
2while_while_cond_35311380___redundant_placeholder16
2while_while_cond_35311380___redundant_placeholder26
2while_while_cond_35311380___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ã
Í
while_cond_35310748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35310748___redundant_placeholder06
2while_while_cond_35310748___redundant_placeholder16
2while_while_cond_35310748___redundant_placeholder26
2while_while_cond_35310748___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
³	

rnn_while_cond_35311850$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1>
:rnn_while_rnn_while_cond_35311850___redundant_placeholder0>
:rnn_while_rnn_while_cond_35311850___redundant_placeholder1>
:rnn_while_rnn_while_cond_35311850___redundant_placeholder2>
:rnn_while_rnn_while_cond_35311850___redundant_placeholder3
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÌJ
¨
A__inference_rnn_layer_call_and_return_conditional_losses_35311274

inputs;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35311189*
condR
while_cond_35311188*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Õ
ð
)__inference_mdnrnn_layer_call_fn_35311625

inputs
unknown:	&
	unknown_0:

	unknown_1:	
	unknown_2:
þ
	unknown_3:	þ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311292p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Ä6
¹
while_body_35312205
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

-__inference_sequential_layer_call_fn_35311059
input_1
unknown:
þ
	unknown_0:	þ
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311052p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ÌJ
¨
A__inference_rnn_layer_call_and_return_conditional_losses_35312434

inputs;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35312349*
condR
while_cond_35312348*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
 
Ö
&__inference_rnn_layer_call_fn_35311972
inputs_0
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311015}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
"
_user_specified_name
inputs/0
Ø9

A__inference_rnn_layer_call_and_return_conditional_losses_35310819

inputs%
lstm_cell_35310736:	&&
lstm_cell_35310738:
!
lstm_cell_35310740:	
identity

identity_1

identity_2¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_maskñ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35310736lstm_cell_35310738lstm_cell_35310740*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310735n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35310736lstm_cell_35310738lstm_cell_35310740*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35310749*
condR
while_cond_35310748*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs
ú

Ô
&__inference_rnn_layer_call_fn_35311987

inputs
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311274u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
¸
Ó
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311517

inputs
rnn_35311500:	& 
rnn_35311502:

rnn_35311504:	'
sequential_35311511:
þ"
sequential_35311513:	þ
identity¢rnn/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall^
rnn/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&¨
rnn/StatefulPartitionedCallStatefulPartitionedCallrnn/Cast:y:0rnn_35311500rnn_35311502rnn_35311504*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311466^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape$rnn/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_35311511sequential_35311513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311089{
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp^rnn/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
å
§
7__inference_mu_logstd_logmix_net_layer_call_fn_35312723

inputs
unknown:
þ
	unknown_0:	þ
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Ô
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311566
input_1
rnn_35311549:	& 
rnn_35311551:

rnn_35311553:	'
sequential_35311560:
þ"
sequential_35311562:	þ
identity¢rnn/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall_
rnn/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&¨
rnn/StatefulPartitionedCallStatefulPartitionedCallrnn/Cast:y:0rnn_35311549rnn_35311551rnn_35311553*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311274^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape$rnn/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_35311560sequential_35311562*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311052{
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp^rnn/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
ÿ
Ð
H__inference_sequential_layer_call_and_return_conditional_losses_35312616

inputsG
3mu_logstd_logmix_net_matmul_readvariableop_resource:
þC
4mu_logstd_logmix_net_biasadd_readvariableop_resource:	þ
identity¢+mu_logstd_logmix_net/BiasAdd/ReadVariableOp¢*mu_logstd_logmix_net/MatMul/ReadVariableOp 
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0¶
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ¡
NoOpNoOp,^mu_logstd_logmix_net/BiasAdd/ReadVariableOp+^mu_logstd_logmix_net/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2Z
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp+mu_logstd_logmix_net/BiasAdd/ReadVariableOp2X
*mu_logstd_logmix_net/MatMul/ReadVariableOp*mu_logstd_logmix_net/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÙZ
¼
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311942

inputs?
,rnn_lstm_cell_matmul_readvariableop_resource:	&B
.rnn_lstm_cell_matmul_1_readvariableop_resource:
<
-rnn_lstm_cell_biasadd_readvariableop_resource:	R
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource:
þN
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource:	þ
identity¢$rnn/lstm_cell/BiasAdd/ReadVariableOp¢#rnn/lstm_cell/MatMul/ReadVariableOp¢%rnn/lstm_cell/MatMul_1/ReadVariableOp¢	rnn/while¢6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp¢5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp^
rnn/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&E
	rnn/ShapeShapernn/Cast:y:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn/transpose	Transposernn/Cast:y:0rnn/transpose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ì
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
#rnn/lstm_cell/MatMul/ReadVariableOpReadVariableOp,rnn_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
rnn/lstm_cell/MatMulMatMulrnn/strided_slice_2:output:0+rnn/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp.rnn_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
rnn/lstm_cell/MatMul_1MatMulrnn/zeros:output:0-rnn/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/addAddV2rnn/lstm_cell/MatMul:product:0 rnn/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$rnn/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp-rnn_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
rnn/lstm_cell/BiasAddBiasAddrnn/lstm_cell/add:z:0,rnn/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
rnn/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ä
rnn/lstm_cell/splitSplit&rnn/lstm_cell/split/split_dim:output:0rnn/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitq
rnn/lstm_cell/SigmoidSigmoidrnn/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_1Sigmoidrnn/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rnn/lstm_cell/mulMulrnn/lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
rnn/lstm_cell/TanhTanhrnn/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_1Mulrnn/lstm_cell/Sigmoid:y:0rnn/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/add_1AddV2rnn/lstm_cell/mul:z:0rnn/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_2Sigmoidrnn/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
rnn/lstm_cell/Tanh_1Tanhrnn/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_2Mulrnn/lstm_cell/Sigmoid_2:y:0rnn/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0,rnn_lstm_cell_matmul_readvariableop_resource.rnn_lstm_cell_matmul_1_readvariableop_resource-rnn_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
rnn_while_body_35311851*#
condR
rnn_while_cond_35311850*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   r
ReshapeReshapernn/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0´
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ³
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0×
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity0sequential/mu_logstd_logmix_net/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ¸
NoOpNoOp%^rnn/lstm_cell/BiasAdd/ReadVariableOp$^rnn/lstm_cell/MatMul/ReadVariableOp&^rnn/lstm_cell/MatMul_1/ReadVariableOp
^rnn/while7^sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6^sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2L
$rnn/lstm_cell/BiasAdd/ReadVariableOp$rnn/lstm_cell/BiasAdd/ReadVariableOp2J
#rnn/lstm_cell/MatMul/ReadVariableOp#rnn/lstm_cell/MatMul/ReadVariableOp2N
%rnn/lstm_cell/MatMul_1/ReadVariableOp%rnn/lstm_cell/MatMul_1/ReadVariableOp2
	rnn/while	rnn/while2p
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp2n
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Ö

G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310886

inputs

states
states_11
matmul_readvariableop_resource:	&4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	&*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
³	

rnn_while_cond_35311699$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1>
:rnn_while_rnn_while_cond_35311699___redundant_placeholder0>
:rnn_while_rnn_while_cond_35311699___redundant_placeholder1>
:rnn_while_rnn_while_cond_35311699___redundant_placeholder2>
:rnn_while_rnn_while_cond_35311699___redundant_placeholder3
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ã
Í
while_cond_35312348
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35312348___redundant_placeholder06
2while_while_cond_35312348___redundant_placeholder16
2while_while_cond_35312348___redundant_placeholder26
2while_while_cond_35312348___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Þ

G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312682

inputs
states_0
states_11
matmul_readvariableop_resource:	&4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	&*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ã
Í
while_cond_35312204
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35312204___redundant_placeholder06
2while_while_cond_35312204___redundant_placeholder16
2while_while_cond_35312204___redundant_placeholder26
2while_while_cond_35312204___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
É	
÷
H__inference_sequential_layer_call_and_return_conditional_losses_35311089

inputs1
mu_logstd_logmix_net_35311083:
þ,
mu_logstd_logmix_net_35311085:	þ
identity¢,mu_logstd_logmix_net/StatefulPartitionedCall§
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_35311083mu_logstd_logmix_net_35311085*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
NoOpNoOp-^mu_logstd_logmix_net/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
;
¹	
rnn_while_body_35311851$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0G
4rnn_while_lstm_cell_matmul_readvariableop_resource_0:	&J
6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0:
D
5rnn_while_lstm_cell_biasadd_readvariableop_resource_0:	
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorE
2rnn_while_lstm_cell_matmul_readvariableop_resource:	&H
4rnn_while_lstm_cell_matmul_1_readvariableop_resource:
B
3rnn_while_lstm_cell_biasadd_readvariableop_resource:	¢*rnn/while/lstm_cell/BiasAdd/ReadVariableOp¢)rnn/while/lstm_cell/MatMul/ReadVariableOp¢+rnn/while/lstm_cell/MatMul_1/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   º
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
)rnn/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp4rnn_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0À
rnn/while/lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:01rnn/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
+rnn/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0§
rnn/while/lstm_cell/MatMul_1MatMulrnn_while_placeholder_23rnn/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
rnn/while/lstm_cell/addAddV2$rnn/while/lstm_cell/MatMul:product:0&rnn/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*rnn/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp5rnn_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
rnn/while/lstm_cell/BiasAddBiasAddrnn/while/lstm_cell/add:z:02rnn/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#rnn/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ö
rnn/while/lstm_cell/splitSplit,rnn/while/lstm_cell/split/split_dim:output:0$rnn/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split}
rnn/while/lstm_cell/SigmoidSigmoid"rnn/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_1Sigmoid"rnn/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mulMul!rnn/while/lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
rnn/while/lstm_cell/TanhTanh"rnn/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_1Mulrnn/while/lstm_cell/Sigmoid:y:0rnn/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/add_1AddV2rnn/while/lstm_cell/mul:z:0rnn/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_2Sigmoid"rnn/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
rnn/while/lstm_cell/Tanh_1Tanhrnn/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_2Mul!rnn/while/lstm_cell/Sigmoid_2:y:0rnn/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identityrnn/while/lstm_cell/mul_2:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/Identity_5Identityrnn/while/lstm_cell/add_1:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
rnn/while/NoOpNoOp+^rnn/while/lstm_cell/BiasAdd/ReadVariableOp*^rnn/while/lstm_cell/MatMul/ReadVariableOp,^rnn/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"l
3rnn_while_lstm_cell_biasadd_readvariableop_resource5rnn_while_lstm_cell_biasadd_readvariableop_resource_0"n
4rnn_while_lstm_cell_matmul_1_readvariableop_resource6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0"j
2rnn_while_lstm_cell_matmul_readvariableop_resource4rnn_while_lstm_cell_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*rnn/while/lstm_cell/BiasAdd/ReadVariableOp*rnn/while/lstm_cell/BiasAdd/ReadVariableOp2V
)rnn/while/lstm_cell/MatMul/ReadVariableOp)rnn/while/lstm_cell/MatMul/ReadVariableOp2Z
+rnn/while/lstm_cell/MatMul_1/ReadVariableOp+rnn/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
;
¹	
rnn_while_body_35310577$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0G
4rnn_while_lstm_cell_matmul_readvariableop_resource_0:	&J
6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0:
D
5rnn_while_lstm_cell_biasadd_readvariableop_resource_0:	
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorE
2rnn_while_lstm_cell_matmul_readvariableop_resource:	&H
4rnn_while_lstm_cell_matmul_1_readvariableop_resource:
B
3rnn_while_lstm_cell_biasadd_readvariableop_resource:	¢*rnn/while/lstm_cell/BiasAdd/ReadVariableOp¢)rnn/while/lstm_cell/MatMul/ReadVariableOp¢+rnn/while/lstm_cell/MatMul_1/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   º
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
)rnn/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp4rnn_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0À
rnn/while/lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:01rnn/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
+rnn/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0§
rnn/while/lstm_cell/MatMul_1MatMulrnn_while_placeholder_23rnn/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
rnn/while/lstm_cell/addAddV2$rnn/while/lstm_cell/MatMul:product:0&rnn/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*rnn/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp5rnn_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
rnn/while/lstm_cell/BiasAddBiasAddrnn/while/lstm_cell/add:z:02rnn/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#rnn/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ö
rnn/while/lstm_cell/splitSplit,rnn/while/lstm_cell/split/split_dim:output:0$rnn/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split}
rnn/while/lstm_cell/SigmoidSigmoid"rnn/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_1Sigmoid"rnn/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mulMul!rnn/while/lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
rnn/while/lstm_cell/TanhTanh"rnn/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_1Mulrnn/while/lstm_cell/Sigmoid:y:0rnn/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/add_1AddV2rnn/while/lstm_cell/mul:z:0rnn/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_2Sigmoid"rnn/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
rnn/while/lstm_cell/Tanh_1Tanhrnn/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_2Mul!rnn/while/lstm_cell/Sigmoid_2:y:0rnn/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identityrnn/while/lstm_cell/mul_2:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/Identity_5Identityrnn/while/lstm_cell/add_1:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
rnn/while/NoOpNoOp+^rnn/while/lstm_cell/BiasAdd/ReadVariableOp*^rnn/while/lstm_cell/MatMul/ReadVariableOp,^rnn/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"l
3rnn_while_lstm_cell_biasadd_readvariableop_resource5rnn_while_lstm_cell_biasadd_readvariableop_resource_0"n
4rnn_while_lstm_cell_matmul_1_readvariableop_resource6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0"j
2rnn_while_lstm_cell_matmul_readvariableop_resource4rnn_while_lstm_cell_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*rnn/while/lstm_cell/BiasAdd/ReadVariableOp*rnn/while/lstm_cell/BiasAdd/ReadVariableOp2V
)rnn/while/lstm_cell/MatMul/ReadVariableOp)rnn/while/lstm_cell/MatMul/ReadVariableOp2Z
+rnn/while/lstm_cell/MatMul_1/ReadVariableOp+rnn/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÿ
Ð
H__inference_sequential_layer_call_and_return_conditional_losses_35312606

inputsG
3mu_logstd_logmix_net_matmul_readvariableop_resource:
þC
4mu_logstd_logmix_net_biasadd_readvariableop_resource:	þ
identity¢+mu_logstd_logmix_net/BiasAdd/ReadVariableOp¢*mu_logstd_logmix_net/MatMul/ReadVariableOp 
*mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp3mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0
mu_logstd_logmix_net/MatMulMatMulinputs2mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
+mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp4mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0¶
mu_logstd_logmix_net/BiasAddBiasAdd%mu_logstd_logmix_net/MatMul:product:03mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
IdentityIdentity%mu_logstd_logmix_net/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ¡
NoOpNoOp,^mu_logstd_logmix_net/BiasAdd/ReadVariableOp+^mu_logstd_logmix_net/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2Z
+mu_logstd_logmix_net/BiasAdd/ReadVariableOp+mu_logstd_logmix_net/BiasAdd/ReadVariableOp2X
*mu_logstd_logmix_net/MatMul/ReadVariableOp*mu_logstd_logmix_net/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
Í
while_cond_35311188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35311188___redundant_placeholder06
2while_while_cond_35311188___redundant_placeholder16
2while_while_cond_35311188___redundant_placeholder26
2while_while_cond_35311188___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ø
ñ
)__inference_mdnrnn_layer_call_fn_35311545
input_1
unknown:	&
	unknown_0:

	unknown_1:	
	unknown_2:
þ
	unknown_3:	þ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311517p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
Ñ

-__inference_sequential_layer_call_fn_35312596

inputs
unknown:
þ
	unknown_0:	þ
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311089p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
ð
)__inference_mdnrnn_layer_call_fn_35311640

inputs
unknown:	&
	unknown_0:

	unknown_1:	
	unknown_2:
þ
	unknown_3:	þ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311517p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Ô

-__inference_sequential_layer_call_fn_35311105
input_1
unknown:
þ
	unknown_0:	þ
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311089p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ä6
¹
while_body_35311189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Þ

G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312714

inputs
states_0
states_11
matmul_readvariableop_resource:	&4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	&*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :º
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitU
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
TanhTanhsplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
mul_1MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
Tanh_1Tanh	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ã
Í
while_cond_35312060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35312060___redundant_placeholder06
2while_while_cond_35312060___redundant_placeholder16
2while_while_cond_35312060___redundant_placeholder26
2while_while_cond_35312060___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÉZ
¿
$__inference__traced_restore_35312898
file_prefix8
%assignvariableop_rnn_lstm_cell_kernel:	&E
1assignvariableop_1_rnn_lstm_cell_recurrent_kernel:
4
%assignvariableop_2_rnn_lstm_cell_bias:	B
.assignvariableop_3_mu_logstd_logmix_net_kernel:
þ;
,assignvariableop_4_mu_logstd_logmix_net_bias:	þ&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: #
assignvariableop_10_total: #
assignvariableop_11_count: B
/assignvariableop_12_adam_rnn_lstm_cell_kernel_m:	&M
9assignvariableop_13_adam_rnn_lstm_cell_recurrent_kernel_m:
<
-assignvariableop_14_adam_rnn_lstm_cell_bias_m:	J
6assignvariableop_15_adam_mu_logstd_logmix_net_kernel_m:
þC
4assignvariableop_16_adam_mu_logstd_logmix_net_bias_m:	þB
/assignvariableop_17_adam_rnn_lstm_cell_kernel_v:	&M
9assignvariableop_18_adam_rnn_lstm_cell_recurrent_kernel_v:
<
-assignvariableop_19_adam_rnn_lstm_cell_bias_v:	J
6assignvariableop_20_adam_mu_logstd_logmix_net_kernel_v:
þC
4assignvariableop_21_adam_mu_logstd_logmix_net_bias_v:	þ
identity_23¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Î

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ô	
valueê	Bç	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp%assignvariableop_rnn_lstm_cell_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_1AssignVariableOp1assignvariableop_1_rnn_lstm_cell_recurrent_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_rnn_lstm_cell_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp.assignvariableop_3_mu_logstd_logmix_net_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_mu_logstd_logmix_net_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_12AssignVariableOp/assignvariableop_12_adam_rnn_lstm_cell_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_13AssignVariableOp9assignvariableop_13_adam_rnn_lstm_cell_recurrent_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp-assignvariableop_14_adam_rnn_lstm_cell_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_15AssignVariableOp6assignvariableop_15_adam_mu_logstd_logmix_net_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_mu_logstd_logmix_net_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_17AssignVariableOp/assignvariableop_17_adam_rnn_lstm_cell_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_18AssignVariableOp9assignvariableop_18_adam_rnn_lstm_cell_recurrent_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp-assignvariableop_19_adam_rnn_lstm_cell_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp6assignvariableop_20_adam_mu_logstd_logmix_net_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_mu_logstd_logmix_net_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ³
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
:  
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ø9

A__inference_rnn_layer_call_and_return_conditional_losses_35311015

inputs%
lstm_cell_35310932:	&&
lstm_cell_35310934:
!
lstm_cell_35310936:	
identity

identity_1

identity_2¢!lstm_cell/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_maskñ
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_35310932lstm_cell_35310934lstm_cell_35310936*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310886n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¼
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_35310932lstm_cell_35310934lstm_cell_35310936*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35310945*
condR
while_cond_35310944*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs
Ì	
ø
H__inference_sequential_layer_call_and_return_conditional_losses_35311114
input_11
mu_logstd_logmix_net_35311108:
þ,
mu_logstd_logmix_net_35311110:	þ
identity¢,mu_logstd_logmix_net/StatefulPartitionedCall¨
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_35311108mu_logstd_logmix_net_35311110*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
NoOpNoOp-^mu_logstd_logmix_net/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ã
Í
while_cond_35310944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35310944___redundant_placeholder06
2while_while_cond_35310944___redundant_placeholder16
2while_while_cond_35310944___redundant_placeholder26
2while_while_cond_35310944___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ø
ñ
)__inference_mdnrnn_layer_call_fn_35311305
input_1
unknown:	&
	unknown_0:

	unknown_1:	
	unknown_2:
þ
	unknown_3:	þ
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311292p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
´
î
&__inference_signature_wrapper_35311610
input_1
unknown:	&
	unknown_0:

	unknown_1:	
	unknown_2:
þ
	unknown_3:	þ
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_35310668p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
¸
Ó
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311292

inputs
rnn_35311275:	& 
rnn_35311277:

rnn_35311279:	'
sequential_35311286:
þ"
sequential_35311288:	þ
identity¢rnn/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall^
rnn/CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&¨
rnn/StatefulPartitionedCallStatefulPartitionedCallrnn/Cast:y:0rnn_35311275rnn_35311277rnn_35311279*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311274^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape$rnn/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_35311286sequential_35311288*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311052{
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp^rnn/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
K
ª
A__inference_rnn_layer_call_and_return_conditional_losses_35312146
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35312061*
condR
while_cond_35312060*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
"
_user_specified_name
inputs/0
»
Ô
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311587
input_1
rnn_35311570:	& 
rnn_35311572:

rnn_35311574:	'
sequential_35311581:
þ"
sequential_35311583:	þ
identity¢rnn/StatefulPartitionedCall¢"sequential/StatefulPartitionedCall_
rnn/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&¨
rnn/StatefulPartitionedCallStatefulPartitionedCallrnn/Cast:y:0rnn_35311570rnn_35311572rnn_35311574*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311466^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
ReshapeReshape$rnn/StatefulPartitionedCall:output:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"sequential/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0sequential_35311581sequential_35311583*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311089{
IdentityIdentity+sequential/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp^rnn/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2:
rnn/StatefulPartitionedCallrnn/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
Ä6
¹
while_body_35312349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ä6
¹
while_body_35311381
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
É	
÷
H__inference_sequential_layer_call_and_return_conditional_losses_35311052

inputs1
mu_logstd_logmix_net_35311046:
þ,
mu_logstd_logmix_net_35311048:	þ
identity¢,mu_logstd_logmix_net/StatefulPartitionedCall§
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinputsmu_logstd_logmix_net_35311046mu_logstd_logmix_net_35311048*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
NoOpNoOp-^mu_logstd_logmix_net/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì	
ø
H__inference_sequential_layer_call_and_return_conditional_losses_35311123
input_11
mu_logstd_logmix_net_35311117:
þ,
mu_logstd_logmix_net_35311119:	þ
identity¢,mu_logstd_logmix_net/StatefulPartitionedCall¨
,mu_logstd_logmix_net/StatefulPartitionedCallStatefulPartitionedCallinput_1mu_logstd_logmix_net_35311117mu_logstd_logmix_net_35311119*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35311045
IdentityIdentity5mu_logstd_logmix_net/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþu
NoOpNoOp-^mu_logstd_logmix_net/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2\
,mu_logstd_logmix_net/StatefulPartitionedCall,mu_logstd_logmix_net/StatefulPartitionedCall:Q M
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ä6
¹
while_body_35312061
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
0while_lstm_cell_matmul_readvariableop_resource_0:	&F
2while_lstm_cell_matmul_1_readvariableop_resource_0:
@
1while_lstm_cell_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
.while_lstm_cell_matmul_readvariableop_resource:	&D
0while_lstm_cell_matmul_1_readvariableop_resource:
>
/while_lstm_cell_biasadd_readvariableop_resource:	¢&while/lstm_cell/BiasAdd/ReadVariableOp¢%while/lstm_cell/MatMul/ReadVariableOp¢'while/lstm_cell/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0´
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ê
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitu
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
while/lstm_cell/TanhTanhwhile/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: w
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/Identity_5Identitywhile/lstm_cell/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ

while/NoOpNoOp'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ú

Ô
&__inference_rnn_layer_call_fn_35312002

inputs
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *U
_output_shapesC
A:ÿÿÿÿÿÿÿÿÿè:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35311466u
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
Ñ

-__inference_sequential_layer_call_fn_35312587

inputs
unknown:
þ
	unknown_0:	þ
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_layer_call_and_return_conditional_losses_35311052p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
Í
while_cond_35312492
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_35312492___redundant_placeholder06
2while_while_cond_35312492___redundant_placeholder16
2while_while_cond_35312492___redundant_placeholder26
2while_while_cond_35312492___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
à	

R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35312733

inputs2
matmul_readvariableop_resource:
þ.
biasadd_readvariableop_resource:	þ
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»Z

#__inference__wrapped_model_35310668
input_1?
,rnn_lstm_cell_matmul_readvariableop_resource:	&B
.rnn_lstm_cell_matmul_1_readvariableop_resource:
<
-rnn_lstm_cell_biasadd_readvariableop_resource:	R
>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource:
þN
?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource:	þ
identity¢$rnn/lstm_cell/BiasAdd/ReadVariableOp¢#rnn/lstm_cell/MatMul/ReadVariableOp¢%rnn/lstm_cell/MatMul_1/ReadVariableOp¢	rnn/while¢6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp¢5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp_
rnn/CastCastinput_1*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&E
	rnn/ShapeShapernn/Cast:y:0*
T0*
_output_shapes
:a
rnn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
rnn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
rnn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
rnn/strided_sliceStridedSlicernn/Shape:output:0 rnn/strided_slice/stack:output:0"rnn/strided_slice/stack_1:output:0"rnn/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
rnn/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros/packedPackrnn/strided_slice:output:0rnn/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:T
rnn/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    y
	rnn/zerosFillrnn/zeros/packed:output:0rnn/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
rnn/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
rnn/zeros_1/packedPackrnn/strided_slice:output:0rnn/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:V
rnn/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
rnn/zeros_1Fillrnn/zeros_1/packed:output:0rnn/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
rnn/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          |
rnn/transpose	Transposernn/Cast:y:0rnn/transpose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&L
rnn/Shape_1Shapernn/transpose:y:0*
T0*
_output_shapes
:c
rnn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
rnn/strided_slice_1StridedSlicernn/Shape_1:output:0"rnn/strided_slice_1/stack:output:0$rnn/strided_slice_1/stack_1:output:0$rnn/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
rnn/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÀ
rnn/TensorArrayV2TensorListReserve(rnn/TensorArrayV2/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9rnn/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ì
+rnn/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn/transpose:y:0Brnn/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒc
rnn/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
rnn/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ý
rnn/strided_slice_2StridedSlicernn/transpose:y:0"rnn/strided_slice_2/stack:output:0$rnn/strided_slice_2/stack_1:output:0$rnn/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
#rnn/lstm_cell/MatMul/ReadVariableOpReadVariableOp,rnn_lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
rnn/lstm_cell/MatMulMatMulrnn/strided_slice_2:output:0+rnn/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%rnn/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp.rnn_lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
rnn/lstm_cell/MatMul_1MatMulrnn/zeros:output:0-rnn/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/addAddV2rnn/lstm_cell/MatMul:product:0 rnn/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$rnn/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp-rnn_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
rnn/lstm_cell/BiasAddBiasAddrnn/lstm_cell/add:z:0,rnn/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
rnn/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ä
rnn/lstm_cell/splitSplit&rnn/lstm_cell/split/split_dim:output:0rnn/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitq
rnn/lstm_cell/SigmoidSigmoidrnn/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_1Sigmoidrnn/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
rnn/lstm_cell/mulMulrnn/lstm_cell/Sigmoid_1:y:0rnn/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
rnn/lstm_cell/TanhTanhrnn/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_1Mulrnn/lstm_cell/Sigmoid:y:0rnn/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/add_1AddV2rnn/lstm_cell/mul:z:0rnn/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
rnn/lstm_cell/Sigmoid_2Sigmoidrnn/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
rnn/lstm_cell/Tanh_1Tanhrnn/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/lstm_cell/mul_2Mulrnn/lstm_cell/Sigmoid_2:y:0rnn/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
!rnn/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
rnn/TensorArrayV2_1TensorListReserve*rnn/TensorArrayV2_1/element_shape:output:0rnn/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒJ
rnn/timeConst*
_output_shapes
: *
dtype0*
value	B : g
rnn/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿX
rnn/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¹
	rnn/whileWhilernn/while/loop_counter:output:0%rnn/while/maximum_iterations:output:0rnn/time:output:0rnn/TensorArrayV2_1:handle:0rnn/zeros:output:0rnn/zeros_1:output:0rnn/strided_slice_1:output:0;rnn/TensorArrayUnstack/TensorListFromTensor:output_handle:0,rnn_lstm_cell_matmul_readvariableop_resource.rnn_lstm_cell_matmul_1_readvariableop_resource-rnn_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *#
bodyR
rnn_while_body_35310577*#
condR
rnn_while_cond_35310576*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
4rnn/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
&rnn/TensorArrayV2Stack/TensorListStackTensorListStackrnn/while:output:3=rnn/TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0l
rnn/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿe
rnn/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: e
rnn/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
rnn/strided_slice_3StridedSlice/rnn/TensorArrayV2Stack/TensorListStack:tensor:0"rnn/strided_slice_3/stack:output:0$rnn/strided_slice_3/stack_1:output:0$rnn/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maski
rnn/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¤
rnn/transpose_1	Transpose/rnn/TensorArrayV2Stack/TensorListStack:tensor:0rnn/transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   r
ReshapeReshapernn/transpose_1:y:0Reshape/shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOpReadVariableOp>sequential_mu_logstd_logmix_net_matmul_readvariableop_resource* 
_output_shapes
:
þ*
dtype0´
&sequential/mu_logstd_logmix_net/MatMulMatMulReshape:output:0=sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ³
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOpReadVariableOp?sequential_mu_logstd_logmix_net_biasadd_readvariableop_resource*
_output_shapes	
:þ*
dtype0×
'sequential/mu_logstd_logmix_net/BiasAddBiasAdd0sequential/mu_logstd_logmix_net/MatMul:product:0>sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity0sequential/mu_logstd_logmix_net/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ¸
NoOpNoOp%^rnn/lstm_cell/BiasAdd/ReadVariableOp$^rnn/lstm_cell/MatMul/ReadVariableOp&^rnn/lstm_cell/MatMul_1/ReadVariableOp
^rnn/while7^sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6^sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ÿÿÿÿÿÿÿÿÿè&: : : : : 2L
$rnn/lstm_cell/BiasAdd/ReadVariableOp$rnn/lstm_cell/BiasAdd/ReadVariableOp2J
#rnn/lstm_cell/MatMul/ReadVariableOp#rnn/lstm_cell/MatMul/ReadVariableOp2N
%rnn/lstm_cell/MatMul_1/ReadVariableOp%rnn/lstm_cell/MatMul_1/ReadVariableOp2
	rnn/while	rnn/while2p
6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp6sequential/mu_logstd_logmix_net/BiasAdd/ReadVariableOp2n
5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp5sequential/mu_logstd_logmix_net/MatMul/ReadVariableOp:U Q
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
!
_user_specified_name	input_1
ô
ö
,__inference_lstm_cell_layer_call_fn_35312633

inputs
states_0
states_1
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310735p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
 
Ö
&__inference_rnn_layer_call_fn_35311957
inputs_0
unknown:	&
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_rnn_layer_call_and_return_conditional_losses_35310819}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
"
_user_specified_name
inputs/0
Ë"
Þ
while_body_35310749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_35310773_0:	&.
while_lstm_cell_35310775_0:
)
while_lstm_cell_35310777_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_35310773:	&,
while_lstm_cell_35310775:
'
while_lstm_cell_35310777:	¢'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0¯
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_35310773_0while_lstm_cell_35310775_0while_lstm_cell_35310777_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35310735Ù
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿv

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_35310773while_lstm_cell_35310773_0"6
while_lstm_cell_35310775while_lstm_cell_35310775_0"6
while_lstm_cell_35310777while_lstm_cell_35310777_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
³	

rnn_while_cond_35310576$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3&
"rnn_while_less_rnn_strided_slice_1>
:rnn_while_rnn_while_cond_35310576___redundant_placeholder0>
:rnn_while_rnn_while_cond_35310576___redundant_placeholder1>
:rnn_while_rnn_while_cond_35310576___redundant_placeholder2>
:rnn_while_rnn_while_cond_35310576___redundant_placeholder3
rnn_while_identity
r
rnn/while/LessLessrnn_while_placeholder"rnn_while_less_rnn_strided_slice_1*
T0*
_output_shapes
: S
rnn/while/IdentityIdentityrnn/while/Less:z:0*
T0
*
_output_shapes
: "1
rnn_while_identityrnn/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÌJ
¨
A__inference_rnn_layer_call_and_return_conditional_losses_35312578

inputs;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35312493*
condR
while_cond_35312492*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
ÌJ
¨
A__inference_rnn_layer_call_and_return_conditional_losses_35311466

inputs;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          n
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:èÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35311381*
condR
while_cond_35311380*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*-
_output_shapes
:èÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿèd
IdentityIdentitytranspose_1:y:0^NoOp*
T0*-
_output_shapes
:ÿÿÿÿÿÿÿÿÿè`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿè&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿè&
 
_user_specified_nameinputs
ÿ4
·

!__inference__traced_save_35312822
file_prefix3
/savev2_rnn_lstm_cell_kernel_read_readvariableop=
9savev2_rnn_lstm_cell_recurrent_kernel_read_readvariableop1
-savev2_rnn_lstm_cell_bias_read_readvariableop:
6savev2_mu_logstd_logmix_net_kernel_read_readvariableop8
4savev2_mu_logstd_logmix_net_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_adam_rnn_lstm_cell_kernel_m_read_readvariableopD
@savev2_adam_rnn_lstm_cell_recurrent_kernel_m_read_readvariableop8
4savev2_adam_rnn_lstm_cell_bias_m_read_readvariableopA
=savev2_adam_mu_logstd_logmix_net_kernel_m_read_readvariableop?
;savev2_adam_mu_logstd_logmix_net_bias_m_read_readvariableop:
6savev2_adam_rnn_lstm_cell_kernel_v_read_readvariableopD
@savev2_adam_rnn_lstm_cell_recurrent_kernel_v_read_readvariableop8
4savev2_adam_rnn_lstm_cell_bias_v_read_readvariableopA
=savev2_adam_mu_logstd_logmix_net_kernel_v_read_readvariableop?
;savev2_adam_mu_logstd_logmix_net_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ë

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ô	
valueê	Bç	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ·

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_rnn_lstm_cell_kernel_read_readvariableop9savev2_rnn_lstm_cell_recurrent_kernel_read_readvariableop-savev2_rnn_lstm_cell_bias_read_readvariableop6savev2_mu_logstd_logmix_net_kernel_read_readvariableop4savev2_mu_logstd_logmix_net_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_adam_rnn_lstm_cell_kernel_m_read_readvariableop@savev2_adam_rnn_lstm_cell_recurrent_kernel_m_read_readvariableop4savev2_adam_rnn_lstm_cell_bias_m_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_m_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_m_read_readvariableop6savev2_adam_rnn_lstm_cell_kernel_v_read_readvariableop@savev2_adam_rnn_lstm_cell_recurrent_kernel_v_read_readvariableop4savev2_adam_rnn_lstm_cell_bias_v_read_readvariableop=savev2_adam_mu_logstd_logmix_net_kernel_v_read_readvariableop;savev2_adam_mu_logstd_logmix_net_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*º
_input_shapes¨
¥: :	&:
::
þ:þ: : : : : : : :	&:
::
þ:þ:	&:
::
þ:þ: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	&:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
þ:!

_output_shapes	
:þ:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	&:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
þ:!

_output_shapes	
:þ:%!

_output_shapes
:	&:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
þ:!

_output_shapes	
:þ:

_output_shapes
: 
K
ª
A__inference_rnn_layer_call_and_return_conditional_losses_35312290
inputs_0;
(lstm_cell_matmul_readvariableop_resource:	&>
*lstm_cell_matmul_1_readvariableop_resource:
8
)lstm_cell_biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢ lstm_cell/BiasAdd/ReadVariableOp¢lstm_cell/MatMul/ReadVariableOp¢!lstm_cell/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
shrink_axis_mask
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes
:	&*
dtype0
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ø
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_spliti
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_cell/TanhTanhlstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell/Tanh_1Tanhlstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_35312205*
condR
while_cond_35312204*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ì
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿl
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
NoOpNoOp!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&: : : 2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&
"
_user_specified_name
inputs/0
;
¹	
rnn_while_body_35311700$
 rnn_while_rnn_while_loop_counter*
&rnn_while_rnn_while_maximum_iterations
rnn_while_placeholder
rnn_while_placeholder_1
rnn_while_placeholder_2
rnn_while_placeholder_3#
rnn_while_rnn_strided_slice_1_0_
[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0G
4rnn_while_lstm_cell_matmul_readvariableop_resource_0:	&J
6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0:
D
5rnn_while_lstm_cell_biasadd_readvariableop_resource_0:	
rnn_while_identity
rnn_while_identity_1
rnn_while_identity_2
rnn_while_identity_3
rnn_while_identity_4
rnn_while_identity_5!
rnn_while_rnn_strided_slice_1]
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensorE
2rnn_while_lstm_cell_matmul_readvariableop_resource:	&H
4rnn_while_lstm_cell_matmul_1_readvariableop_resource:
B
3rnn_while_lstm_cell_biasadd_readvariableop_resource:	¢*rnn/while/lstm_cell/BiasAdd/ReadVariableOp¢)rnn/while/lstm_cell/MatMul/ReadVariableOp¢+rnn/while/lstm_cell/MatMul_1/ReadVariableOp
;rnn/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ&   º
-rnn/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0rnn_while_placeholderDrnn/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ&*
element_dtype0
)rnn/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp4rnn_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes
:	&*
dtype0À
rnn/while/lstm_cell/MatMulMatMul4rnn/while/TensorArrayV2Read/TensorListGetItem:item:01rnn/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
+rnn/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0§
rnn/while/lstm_cell/MatMul_1MatMulrnn_while_placeholder_23rnn/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
rnn/while/lstm_cell/addAddV2$rnn/while/lstm_cell/MatMul:product:0&rnn/while/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*rnn/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp5rnn_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0ª
rnn/while/lstm_cell/BiasAddBiasAddrnn/while/lstm_cell/add:z:02rnn/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
#rnn/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ö
rnn/while/lstm_cell/splitSplit,rnn/while/lstm_cell/split/split_dim:output:0$rnn/while/lstm_cell/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split}
rnn/while/lstm_cell/SigmoidSigmoid"rnn/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_1Sigmoid"rnn/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mulMul!rnn/while/lstm_cell/Sigmoid_1:y:0rnn_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
rnn/while/lstm_cell/TanhTanh"rnn/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_1Mulrnn/while/lstm_cell/Sigmoid:y:0rnn/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/add_1AddV2rnn/while/lstm_cell/mul:z:0rnn/while/lstm_cell/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/Sigmoid_2Sigmoid"rnn/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
rnn/while/lstm_cell/Tanh_1Tanhrnn/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/lstm_cell/mul_2Mul!rnn/while/lstm_cell/Sigmoid_2:y:0rnn/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
.rnn/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemrnn_while_placeholder_1rnn_while_placeholderrnn/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒQ
rnn/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
rnn/while/addAddV2rnn_while_placeholderrnn/while/add/y:output:0*
T0*
_output_shapes
: S
rnn/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
rnn/while/add_1AddV2 rnn_while_rnn_while_loop_counterrnn/while/add_1/y:output:0*
T0*
_output_shapes
: e
rnn/while/IdentityIdentityrnn/while/add_1:z:0^rnn/while/NoOp*
T0*
_output_shapes
: z
rnn/while/Identity_1Identity&rnn_while_rnn_while_maximum_iterations^rnn/while/NoOp*
T0*
_output_shapes
: e
rnn/while/Identity_2Identityrnn/while/add:z:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_3Identity>rnn/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^rnn/while/NoOp*
T0*
_output_shapes
: 
rnn/while/Identity_4Identityrnn/while/lstm_cell/mul_2:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
rnn/while/Identity_5Identityrnn/while/lstm_cell/add_1:z:0^rnn/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
rnn/while/NoOpNoOp+^rnn/while/lstm_cell/BiasAdd/ReadVariableOp*^rnn/while/lstm_cell/MatMul/ReadVariableOp,^rnn/while/lstm_cell/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "1
rnn_while_identityrnn/while/Identity:output:0"5
rnn_while_identity_1rnn/while/Identity_1:output:0"5
rnn_while_identity_2rnn/while/Identity_2:output:0"5
rnn_while_identity_3rnn/while/Identity_3:output:0"5
rnn_while_identity_4rnn/while/Identity_4:output:0"5
rnn_while_identity_5rnn/while/Identity_5:output:0"l
3rnn_while_lstm_cell_biasadd_readvariableop_resource5rnn_while_lstm_cell_biasadd_readvariableop_resource_0"n
4rnn_while_lstm_cell_matmul_1_readvariableop_resource6rnn_while_lstm_cell_matmul_1_readvariableop_resource_0"j
2rnn_while_lstm_cell_matmul_readvariableop_resource4rnn_while_lstm_cell_matmul_readvariableop_resource_0"@
rnn_while_rnn_strided_slice_1rnn_while_rnn_strided_slice_1_0"¸
Yrnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor[rnn_while_tensorarrayv2read_tensorlistgetitem_rnn_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*rnn/while/lstm_cell/BiasAdd/ReadVariableOp*rnn/while/lstm_cell/BiasAdd/ReadVariableOp2V
)rnn/while/lstm_cell/MatMul/ReadVariableOp)rnn/while/lstm_cell/MatMul/ReadVariableOp2Z
+rnn/while/lstm_cell/MatMul_1/ReadVariableOp+rnn/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: "µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
@
input_15
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿè&=
output_11
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿþtensorflow/serving/predict:ÂÆ

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	inference_base

out_net

signatures"
_tf_keras_model
C
0
1
2
3
4"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Í
trace_0
trace_1
trace_2
trace_32â
)__inference_mdnrnn_layer_call_fn_35311305
)__inference_mdnrnn_layer_call_fn_35311625
)__inference_mdnrnn_layer_call_fn_35311640
)__inference_mdnrnn_layer_call_fn_35311545³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
¹
trace_0
trace_1
trace_2
trace_32Î
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311791
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311942
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311566
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311587³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ÎBË
#__inference__wrapped_model_35310668input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­
iter

beta_1

 beta_2
	!decay
"learning_ratemqmrmsmtmuvvvwvxvyvz"
	optimizer
Ã
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)cell
*
state_spec"
_tf_keras_rnn_layer
Ñ
+layer_with_weights-0
+layer-0
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_sequential
,
2serving_default"
signature_map
':%	&2rnn/lstm_cell/kernel
2:0
2rnn/lstm_cell/recurrent_kernel
!:2rnn/lstm_cell/bias
/:-
þ2mu_logstd_logmix_net/kernel
(:&þ2mu_logstd_logmix_net/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ïBì
)__inference_mdnrnn_layer_call_fn_35311305input_1"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
)__inference_mdnrnn_layer_call_fn_35311625inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
îBë
)__inference_mdnrnn_layer_call_fn_35311640inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ïBì
)__inference_mdnrnn_layer_call_fn_35311545input_1"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311791inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311942inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311566input_1"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311587input_1"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

4states
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ó
:trace_0
;trace_1
<trace_2
=trace_32
&__inference_rnn_layer_call_fn_35311957
&__inference_rnn_layer_call_fn_35311972
&__inference_rnn_layer_call_fn_35311987
&__inference_rnn_layer_call_fn_35312002å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z:trace_0z;trace_1z<trace_2z=trace_3
ß
>trace_0
?trace_1
@trace_2
Atrace_32ô
A__inference_rnn_layer_call_and_return_conditional_losses_35312146
A__inference_rnn_layer_call_and_return_conditional_losses_35312290
A__inference_rnn_layer_call_and_return_conditional_losses_35312434
A__inference_rnn_layer_call_and_return_conditional_losses_35312578å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z>trace_0z?trace_1z@trace_2zAtrace_3
ø
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
I
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
»
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
é
Utrace_0
Vtrace_1
Wtrace_2
Xtrace_32þ
-__inference_sequential_layer_call_fn_35311059
-__inference_sequential_layer_call_fn_35312587
-__inference_sequential_layer_call_fn_35312596
-__inference_sequential_layer_call_fn_35311105¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zUtrace_0zVtrace_1zWtrace_2zXtrace_3
Õ
Ytrace_0
Ztrace_1
[trace_2
\trace_32ê
H__inference_sequential_layer_call_and_return_conditional_losses_35312606
H__inference_sequential_layer_call_and_return_conditional_losses_35312616
H__inference_sequential_layer_call_and_return_conditional_losses_35311114
H__inference_sequential_layer_call_and_return_conditional_losses_35311123¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zYtrace_0zZtrace_1z[trace_2z\trace_3
ÍBÊ
&__inference_signature_wrapper_35311610input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
]	variables
^	keras_api
	_total
	`count"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
)0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
&__inference_rnn_layer_call_fn_35311957inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_rnn_layer_call_fn_35311972inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_rnn_layer_call_fn_35311987inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
&__inference_rnn_layer_call_fn_35312002inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ºB·
A__inference_rnn_layer_call_and_return_conditional_losses_35312146inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ºB·
A__inference_rnn_layer_call_and_return_conditional_losses_35312290inputs/0"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_rnn_layer_call_and_return_conditional_losses_35312434inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
A__inference_rnn_layer_call_and_return_conditional_losses_35312578inputs"å
Ü²Ø
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ó
ftrace_0
gtrace_12
,__inference_lstm_cell_layer_call_fn_35312633
,__inference_lstm_cell_layer_call_fn_35312650½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zftrace_0zgtrace_1

htrace_0
itrace_12Ò
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312682
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312714½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zhtrace_0zitrace_1
"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
û
otrace_02Þ
7__inference_mu_logstd_logmix_net_layer_call_fn_35312723¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zotrace_0

ptrace_02ù
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35312733¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zptrace_0
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÿBü
-__inference_sequential_layer_call_fn_35311059input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_layer_call_fn_35312587inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þBû
-__inference_sequential_layer_call_fn_35312596inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
-__inference_sequential_layer_call_fn_35311105input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_layer_call_and_return_conditional_losses_35312606inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_layer_call_and_return_conditional_losses_35312616inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_layer_call_and_return_conditional_losses_35311114input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
H__inference_sequential_layer_call_and_return_conditional_losses_35311123input_1"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
.
_0
`1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_lstm_cell_layer_call_fn_35312633inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
,__inference_lstm_cell_layer_call_fn_35312650inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312682inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ªB§
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312714inputsstates/0states/1"½
´²°
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ëBè
7__inference_mu_logstd_logmix_net_layer_call_fn_35312723inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35312733inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,:*	&2Adam/rnn/lstm_cell/kernel/m
7:5
2%Adam/rnn/lstm_cell/recurrent_kernel/m
&:$2Adam/rnn/lstm_cell/bias/m
4:2
þ2"Adam/mu_logstd_logmix_net/kernel/m
-:+þ2 Adam/mu_logstd_logmix_net/bias/m
,:*	&2Adam/rnn/lstm_cell/kernel/v
7:5
2%Adam/rnn/lstm_cell/recurrent_kernel/v
&:$2Adam/rnn/lstm_cell/bias/v
4:2
þ2"Adam/mu_logstd_logmix_net/kernel/v
-:+þ2 Adam/mu_logstd_logmix_net/bias/v
#__inference__wrapped_model_35310668t5¢2
+¢(
&#
input_1ÿÿÿÿÿÿÿÿÿè&
ª "4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿþÎ
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312682¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ&
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Î
G__inference_lstm_cell_layer_call_and_return_conditional_losses_35312714¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ&
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 £
,__inference_lstm_cell_layer_call_fn_35312633ò¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ&
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ£
,__inference_lstm_cell_layer_call_fn_35312650ò¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ&
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ²
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311566j9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿè&
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ²
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311587j9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿè&
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ±
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311791i8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿè&
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ±
D__inference_mdnrnn_layer_call_and_return_conditional_losses_35311942i8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿè&
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 
)__inference_mdnrnn_layer_call_fn_35311305]9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿè&
p 
ª "ÿÿÿÿÿÿÿÿÿþ
)__inference_mdnrnn_layer_call_fn_35311545]9¢6
/¢,
&#
input_1ÿÿÿÿÿÿÿÿÿè&
p
ª "ÿÿÿÿÿÿÿÿÿþ
)__inference_mdnrnn_layer_call_fn_35311625\8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿè&
p 
ª "ÿÿÿÿÿÿÿÿÿþ
)__inference_mdnrnn_layer_call_fn_35311640\8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿè&
p
ª "ÿÿÿÿÿÿÿÿÿþ´
R__inference_mu_logstd_logmix_net_layer_call_and_return_conditional_losses_35312733^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 
7__inference_mu_logstd_logmix_net_layer_call_fn_35312723Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿþ
A__inference_rnn_layer_call_and_return_conditional_losses_35312146ÖS¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&

 
p 

 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_rnn_layer_call_and_return_conditional_losses_35312290ÖS¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&

 
p

 

 
ª "z¢w
pm
+(
0/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_rnn_layer_call_and_return_conditional_losses_35312434¿D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿè&

 
p 

 

 
ª "r¢o
he
# 
0/0ÿÿÿÿÿÿÿÿÿè

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
A__inference_rnn_layer_call_and_return_conditional_losses_35312578¿D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿè&

 
p

 

 
ª "r¢o
he
# 
0/0ÿÿÿÿÿÿÿÿÿè

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 ñ
&__inference_rnn_layer_call_fn_35311957ÆS¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&

 
p 

 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿñ
&__inference_rnn_layer_call_fn_35311972ÆS¢P
I¢F
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ&

 
p

 

 
ª "jg
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÚ
&__inference_rnn_layer_call_fn_35311987¯D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿè&

 
p 

 

 
ª "b_
!
0ÿÿÿÿÿÿÿÿÿè

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÚ
&__inference_rnn_layer_call_fn_35312002¯D¢A
:¢7
%"
inputsÿÿÿÿÿÿÿÿÿè&

 
p

 

 
ª "b_
!
0ÿÿÿÿÿÿÿÿÿè

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿ³
H__inference_sequential_layer_call_and_return_conditional_losses_35311114g9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ³
H__inference_sequential_layer_call_and_return_conditional_losses_35311123g9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ²
H__inference_sequential_layer_call_and_return_conditional_losses_35312606f8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 ²
H__inference_sequential_layer_call_and_return_conditional_losses_35312616f8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿþ
 
-__inference_sequential_layer_call_fn_35311059Z9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿþ
-__inference_sequential_layer_call_fn_35311105Z9¢6
/¢,
"
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿþ
-__inference_sequential_layer_call_fn_35312587Y8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿþ
-__inference_sequential_layer_call_fn_35312596Y8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿþ©
&__inference_signature_wrapper_35311610@¢=
¢ 
6ª3
1
input_1&#
input_1ÿÿÿÿÿÿÿÿÿè&"4ª1
/
output_1# 
output_1ÿÿÿÿÿÿÿÿÿþ