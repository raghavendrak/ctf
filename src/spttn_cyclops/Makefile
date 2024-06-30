LOBJS = prepare_kernel.o execute_kernel.o 
OBJS = $(addprefix $(ODIR)/, $(LOBJS))

#%d | r ! grep -ho "\.\..*\.h" *.cxx *.h | sort | uniq
HDRS = ../../Makefile $(BDIR)/config.mk  ../interface/functions.h ../contraction/contraction.h ../mapping/distribution.h ../mapping/mapping.h ../redistribution/nosym_transp.h ../redistribution/redist.h ../scaling/strp_tsr.h ../shared/iter_tsr.h ../shared/memcontrol.h ../shared/offload.h ../shared/util.h ../symmetry/sym_indices.h ../symmetry/symmetrization.h ../tensor/algstrct.h ../tensor/untyped_tensor.h ../shared/model.h ../shared/init_models.h ../sparse_formats/coo.h ../sparse_formats/csr.h
 
ctf: $(OBJS) 

$(OBJS): $(ODIR)/%.o: %.cxx *.h  $(HDRS)
	$(FCXX) -c $< -o $@


