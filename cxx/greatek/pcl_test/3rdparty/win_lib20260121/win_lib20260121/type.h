#pragma once

typedef struct
{
	float x;
	float y;
	float z;
	int r;
	int g;
	int b;
}pointxyzrgb;

struct Point2i {
	int x;
	int y;
};

enum class PixelFormat {
	GRAY8,
	RGB8,
	BGR8,
	RGBA8,
	DEPTH16,
	FLOAT32,
	XYZ32
};

struct ImageData {
	uint8_t* data = nullptr;     // 图像像素数据
	int width = 0;
	int height = 0;
	int channels = 0;
	int stride = 0;
	PixelFormat format = PixelFormat::BGR8;
};
