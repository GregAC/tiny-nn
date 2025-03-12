VERILATOR_ARGS = --binary --trace-fst --trace-structs

RTL_FILES = ./rtl/tiny_nn_pkg.sv \
  ./rtl/fp_add.sv \
  ./rtl/fp_mul.sv \
	./rtl/fp_cmp.sv \
  ./rtl/tiny_nn_core.sv \
  ./rtl/tiny_nn_top.sv \

TOP_DV_FILES = ./dv/tiny_nn_top_tb.sv

FP_ADD_DV_FILES = ./dv/fp_op_tester.sv \
  ./dv/fp_add_tb.sv

FP_MUL_DV_FILES = ./dv/fp_op_tester.sv \
  ./dv/fp_mul_tb.sv

SV2V_OUT = ./sv2v_out/tiny_nn.v

top_dv: $(RTL_FILES) $(TOP_DV_FILES)
	verilator $(VERILATOR_ARGS) \
		--top-module tiny_nn_top_tb \
		$?

fp_add_dv: $(RTL_FILES) $(FP_ADD_DV_FILES)
	verilator $(VERILATOR_ARGS) \
		--top-module fp_add_tb \
	  $?

fp_mul_dv: $(RTL_FILES) $(FP_MUL_DV_FILES)
	verilator $(VERILATOR_ARGS) \
		--top-module fp_mul_tb \
	  $?

top_dv_sv2v: $(SV2V_OUT) $(TOP_DV_FILES)
	verilator $(VERILATOR_ARGS) \
		-Wno-fatal \
		--top-module tiny_nn_top_tb \
		-o Vtiny_nn_top_tb_sv2v \
		$?

$(SV2V_OUT): $(RTL_FILES)
	mkdir -p $(dir $(SV2V_OUT))
	sv2v $? > $(SV2V_OUT)

sv2v: $(SV2V_OUT)

clean:
	rm -rf ./obj_dir
	rm -f $(SV2V_OUT)

all: top_dv fp_add_dv fp_mul_dv sv2v top_dv_sv2v

.PHONY: top_dv fp_add_dv fp_mul_dv sv2v top_dv_sv2v all clean
