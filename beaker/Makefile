# Test Makefile for beaker Make compatibility
# Variable definition: path to beaker executable
BEAKER = ./target/release/beaker

# Declare phony targets (not real files)
.PHONY: all clean

# Default target: build both crop and cutout outputs
all: example_crop.jpg example_cutout.png

# Rule to generate cropped bird image from input
example_crop.jpg: example.jpg
	# Run beaker detect with crop, metadata, and depfile generation
	$(BEAKER) detect $< --crop head --metadata --depfile $@.d
# Include dependency file (ignore if missing due to -)
-include example_crop.jpg.d

# Rule to generate cutout image from input
example_cutout.png: example.jpg
	# Run beaker cutout with metadata and depfile generation
	$(BEAKER) cutout $< --metadata --depfile $@.d
# Include dependency file (ignore if missing due to -)
-include example_cutout.png.d

# Clean up all generated files
clean:
	# Remove output images, metadata, and dependency files
	rm -f example_crop.jpg example_cutout.png example_mask.png *.beaker.toml *.d
	# Remove stamp cache to force complete rebuild
	rm -rf ~/.cache/beaker/stamps/
