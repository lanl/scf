
# paths
bindir = $(PWD)
objdir = ./obj
moddir = ./obj

# dependencies
object1 = generate_data.o
object2 = generate_data_finetune.o
obj1 = $(addprefix $(objdir)/, $(object1))
obj2 = $(addprefix $(objdir)/, $(object2))

# targets
exec1 = $(PWD)/generate_data
exec2 = $(PWD)/generate_data_finetune

all: makedir $(exec1) $(exec2)

# options
flitdir = $(HOME)/src/libflit
rgmdir = $(HOME)/src/rgm
include $(flitdir)/src/Makefile.in
inc = $(base_inc) \
	-I$(flitdir)/lib \
	-I$(rgmdir)/lib
lflags = $(rgmdir)/lib/librgm.a \
	$(flitdir)/lib/libflit.a \
	$(base_lflags)
fflags = $(base_fflags) \
	#-O0 -g -check bounds

# compile
$(objdir)/%.o : ./%.f90
	$(fc) -o $@ -c $(fflags) $(inc) $<

# link
$(exec1) : $(obj1)
	$(fc) -o $@ $^ $(lflags) $(inc)

$(exec2) : $(obj2)
	$(fc) -o $@ $^ $(lflags) $(inc)

# make directory
makedir:
	-@mkdir -p $(bindir)
	-@mkdir -p $(objdir)
	-@mkdir -p $(moddir)

# clean
clean:
	-@rm -rf $(objdir)/*.o $(moddir)/*.mod
