#ifndef METADATA_H_
#define METADATA_H_

/* Struct for storing an array of character structs.
   Intended to make it simpler to output metadata
   through the C API. */

struct Metadata {
	MetadataItem* items;
	int num_items;
};

// Stores each individual character, along with its timing information
struct MetadataItem {
	char* character;
	int timestep;
	float start_time;
};

#endif  // METADATA_H_
