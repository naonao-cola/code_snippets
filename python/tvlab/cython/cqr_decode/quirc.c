/* quirc -- QR-code recognition library
 * Copyright (C) 2010-2012 Daniel Beer <dlbeer@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include "quirc_internal.h"

const char *quirc_version(void)
{
	return "1.0";
}

struct quirc *quirc_new(void)
{
	struct quirc *q = malloc(sizeof(*q));

	if (!q)
		return NULL;

	memset(q, 0, sizeof(*q));
	return q;
}

void quirc_destroy(struct quirc *q)
{
	free(q->image);
	/* q->pixels may alias q->image when their type representation is of the
	   same size, so we need to be careful here to avoid a double free */
	if (!QUIRC_PIXEL_ALIAS_IMAGE)
		free(q->pixels);
	free(q);
}

int quirc_resize(struct quirc *q, int w, int h)
{
	uint8_t		*image  = NULL;
	quirc_pixel_t	*pixels = NULL;

	/*
	 * XXX: w and h should be size_t (or at least unsigned) as negatives
	 * values would not make much sense. The downside is that it would break
	 * both the API and ABI. Thus, at the moment, let's just do a sanity
	 * check.
	 */
	if (w < 0 || h < 0)
		goto fail;

	/*
	 * alloc a new buffer for q->image. We avoid realloc(3) because we want
	 * on failure to be leave `q` in a consistant, unmodified state.
	 */
	image = calloc(w, h);
	if (!image)
		goto fail;

	/* compute the "old" (i.e. currently allocated) and the "new"
	   (i.e. requested) image dimensions */
	size_t olddim = q->w * q->h;
	size_t newdim = w * h;
	size_t min = (olddim < newdim ? olddim : newdim);

	/*
	 * copy the data into the new buffer, avoiding (a) to read beyond the
	 * old buffer when the new size is greater and (b) to write beyond the
	 * new buffer when the new size is smaller, hence the min computation.
	 */
	(void)memcpy(image, q->image, min);

	/* alloc a new buffer for q->pixels if needed */
	if (!QUIRC_PIXEL_ALIAS_IMAGE) {
		pixels = calloc(newdim, sizeof(quirc_pixel_t));
		if (!pixels)
			goto fail;
	}

	/* alloc succeeded, update `q` with the new size and buffers */
	q->w = w;
	q->h = h;
	free(q->image);
	q->image = image;
	if (!QUIRC_PIXEL_ALIAS_IMAGE) {
		free(q->pixels);
		q->pixels = pixels;
	}

	return 0;
	/* NOTREACHED */
fail:
	free(image);
	free(pixels);

	return -1;
}

int quirc_count(const struct quirc *q)
{
	return q->num_grids;
}

static const char *const error_table[] = {
	[QUIRC_SUCCESS] = "Success",
	[QUIRC_ERROR_INVALID_GRID_SIZE] = "Invalid grid size",
	[QUIRC_ERROR_INVALID_VERSION] = "Invalid version",
	[QUIRC_ERROR_FORMAT_ECC] = "Format data ECC failure",
	[QUIRC_ERROR_DATA_ECC] = "ECC failure",
	[QUIRC_ERROR_UNKNOWN_DATA_TYPE] = "Unknown data type",
	[QUIRC_ERROR_DATA_OVERFLOW] = "Data overflow",
	[QUIRC_ERROR_DATA_UNDERFLOW] = "Data underflow"
};

const char *quirc_strerror(quirc_decode_error_t err)
{
	if (err >= 0 && err < sizeof(error_table) / sizeof(error_table[0]))
		return error_table[err];

	return "Unknown error";
}

void *create_quirc(unsigned char *img_ptr, int w, int h,
        int32_t *qr_infos_ptr, int *id_cnt)
{
    struct quirc *q = quirc_new();
    quirc_resize(q, w, h);
    uint8_t* dst_ptr = quirc_begin(q, NULL, NULL);
    memcpy(dst_ptr, img_ptr, w * h);
    if (qr_infos_ptr != NULL) {
        quirc_end_skip_detect(q, qr_infos_ptr, *id_cnt);
    } else {
        quirc_end(q);
    }
    *id_cnt = quirc_count(q);
    return (void*)q;
}

void destory_quirc(void *q)
{
    quirc_destroy((struct quirc*)q);
}

void quirc_extract_decode(void *q, int index, int *size, int *ecc_level,
                          int *mask, int *data_type, int *eci,
                          unsigned char* payload, int *payload_len, int *xy,
                          int *score, int *ecc_rate,
                          char* cell_bitmap, char *err_desc)
{
    struct quirc_code code;
    struct quirc_data data;
    quirc_decode_error_t err;

    quirc_extract((struct quirc*)q, index, &code);
    xy[0] = code.corners[0].x;
    xy[1] = code.corners[0].y;
    xy[2] = code.corners[1].x;
    xy[3] = code.corners[1].y;
    xy[4] = code.corners[2].x;
    xy[5] = code.corners[2].y;
    xy[6] = code.corners[3].x;
    xy[7] = code.corners[3].y;
    *score = code.score;
    *size = code.size;
    if (cell_bitmap != NULL) {
        int u, v;
        memset(cell_bitmap, 1, code.size*code.size);
        for (v = 0; v < code.size; v++) {
            for (u = 0; u < code.size; u++) {
                int p = v * code.size + u;
                int v = code.cell_bitmap[p >> 3] & (1 << (p & 7));
                if (v) cell_bitmap[p] = 0;
            }
        }
    }

    err = quirc_decode(&code, &data);
    if (!err) {
        *ecc_level = data.ecc_level;
        *ecc_rate = data.ecc_rate;
        memcpy(payload, data.payload, data.payload_len);
        payload[data.payload_len] = 0;
        *payload_len = data.payload_len;
        *mask = data.mask;
        *data_type = data.data_type;
        *eci = data.eci;
    } else {
        const char * err_str = quirc_strerror(err);
        strcpy(err_desc, err_str);
    }
}
