#include <opencv2/opencv.hpp>
//#include <iostream>
// using namespace std;
// The class Mat_ represents an n-dimensional dense numerical single-channel or multi-channel array
template<typename _Tp, int chs> class Mat_ {
public:
	typedef _Tp value_type;

	// default constructor
	Mat_() : rows(0), cols(0), channels(0), data(NULL), step(0), allocated(false), datastart(NULL), dataend(NULL) {}
	// constructs 2D matrix of the specified size
	Mat_(int _rows, int _cols);
	// constucts 2D matrix and fills it with the specified value _s
	Mat_(int _rows, int _cols, const Scalar& _s);
	// constructor for matrix headers pointing to user-allocated data, no data is copied
	Mat_(int _rows, int _cols, void* _data);
	// copy constructor, NOTE: deep copy
	Mat_(const Mat_<_Tp, chs>& _m);
	Mat_& operator = (const Mat_& _m);

	// reports whether the matrix is continuous or not
	bool isContinuous() const;
	// returns true if the matrix is a submatrix of another matrix
	bool isSubmatrix() const;

	// copies the matrix content to "_m"
	void copyTo(Mat_<_Tp, chs>& _m, const Rect& rect = Rect(0, 0, 0, 0)) const;

	// return typed pointer to the specified matrix row,i0, A 0-based row index
	const uchar* ptr(int i0 = 0) const;
	uchar* ptr(int i0 = 0);

	// no data is copied, no memory is allocated
	void getROI(Mat_<_Tp, chs>& _m, const Rect& rect = Rect(0, 0, 0, 0));
	// Locates the matrix header within a parent matrix
	void locateROI(Size& wholeSize, Point& ofs) const;
	// Adjusts a submatrix size and position within the parent matrix
	void adjustROI(int dtop, int dbottom, int dleft, int dright);

	// value converted to the actual array type
	void setTo(const Scalar& _value);

	// Converts an array to another data type with optional scaling
	// the method converts source pixel values to the target data type
	// if it does not have a proper size before the operation, it is reallocated
	// \f[m(x,y) = saturate \_ cast<rType>( \alpha (*this)(x,y) +  \beta )\f]
	template<typename _Tp2>
	void convertTo(Mat_<_Tp2, chs>& _m, double alpha = 1, const Scalar& scalar = Scalar(0, 0, 0, 0)) const;

	Mat_<_Tp, chs>& zeros(int _rows, int _cols);

	// returns the matrix cols and rows
	Size size() const;
	// returns true if Mat_::total() is 0 or if Mat_::data is NULL
	bool empty() const;
	// returns the matrix element size in bytes: sizeof(_Tp) * channels
	size_t elemSize() const;
	// returns the size of each matrix element channel in bytes: sizeof(_Tp)
	size_t elemSize1() const;
	// returns the total number of array elements
	size_t total() const;

	// release memory
	inline void release();
	// destructor - calls release()
	~Mat_() { release(); };

public:
	// the number of rows and columns
	int rows, cols;
	// channel num
	int channels;
	// pointer to the data
	uchar* data;
	// bytes per row
	int step; // stride
	// memory allocation flag
	bool allocated;
	// helper fields used in locateROI and adjustROI
	const uchar* datastart;
	const uchar* dataend;
}; // Mat_

typedef Mat_<uchar, 1> Mat1Gray;
typedef Mat_<uchar, 3> Mat3BGR;
typedef Mat_<uchar, 4> Mat4BGRA;

//////////////////////////////// Size_ ////////////////////////////////
// Template class for specifying the size of an image or rectangle
template<typename _Tp> class Size_
{
public:
	typedef _Tp value_type;

	//! various constructors
	Size_();
	Size_(_Tp _width, _Tp _height);
	Size_(const Size_& sz);
	Size_(const Point_<_Tp>& pt);

	Size_& operator = (const Size_& sz);
	//! the area (width*height)
	_Tp area() const;

	//! conversion of another data type.
	template<typename _Tp2> operator Size_<_Tp2>() const;

	_Tp width, height; // the width and the height
};

typedef Size_<int> Size2i;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;


static inline int fbcRound(float value)
{
	// it's ok if round does not comply with IEEE754 standard;
	// it should allow +/-1 difference when the other functions use round
	return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


static inline int fbcFloor(float value)
{
	int i = fbcRound(value);
	float diff = (float)(value - i);
	return i - (diff < 0);
}



template<typename _Tp, int chs>
static int resize_cubic(const Mat_<_Tp, chs>& src, Mat_<_Tp, chs>& dst)
{
	Size ssize = src.size();
	Size dsize = dst.size();

	double inv_scale_x = (double)dsize.width / ssize.width;
	double inv_scale_y = (double)dsize.height / ssize.height;
	double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;

	int cn = dst.channels;
	int k, sx, sy, dx, dy;
	int xmin = 0, xmax = dsize.width, width = dsize.width*cn;
	bool fixpt = sizeof(_Tp) == 1 ? true : false;
	float fx, fy;
	int ksize = 4, ksize2;
	ksize2 = ksize / 2;

	AutoBuffer<uchar> _buffer((width + dsize.height)*(sizeof(int) + sizeof(float)*ksize));
	int* xofs = (int*)(uchar*)_buffer;
	int* yofs = xofs + width;
	float* alpha = (float*)(yofs + dsize.height);
	short* ialpha = (short*)alpha;
	float* beta = alpha + width*ksize;
	short* ibeta = ialpha + width*ksize;
	float cbuf[MAX_ESIZE];

	for (dx = 0; dx < dsize.width; dx++) {
		fx = (float)((dx + 0.5)*scale_x - 0.5);
		sx = fbcFloor(fx);
		fx -= sx;

		if (sx < ksize2 - 1) {
			xmin = dx + 1;
		}

		if (sx + ksize2 >= ssize.width) {
			xmax = std::min(xmax, dx);
		}

		for (k = 0, sx *= cn; k < cn; k++) {
			xofs[dx*cn + k] = sx + k;
		}

		interpolateCubic<float>(fx, cbuf);

		if (fixpt) {
			for (k = 0; k < ksize; k++) {
				ialpha[dx*cn*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
			}
			for (; k < cn*ksize; k++) {
				ialpha[dx*cn*ksize + k] = ialpha[dx*cn*ksize + k - ksize];
			}
		} else {
			for (k = 0; k < ksize; k++) {
				alpha[dx*cn*ksize + k] = cbuf[k];
			}
			for (; k < cn*ksize; k++) {
				alpha[dx*cn*ksize + k] = alpha[dx*cn*ksize + k - ksize];
			}
		}
	}

	for (dy = 0; dy < dsize.height; dy++) {
		fy = (float)((dy + 0.5)*scale_y - 0.5);
		sy = fbcFloor(fy);
		fy -= sy;

		yofs[dy] = sy;
		interpolateCubic<float>(fy, cbuf);

		if (fixpt) {
			for (k = 0; k < ksize; k++) {
				ibeta[dy*ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
			}
		} else {
			for (k = 0; k < ksize; k++) {
				beta[dy*ksize + k] = cbuf[k];
			}
		}
	}

	if (sizeof(_Tp) == 1) { // uchar
		typedef uchar value_type; // HResizeCubic/VResizeCubic
		typedef int buf_type;
		typedef short alpha_type;

		resizeGeneric_Cubic<_Tp, value_type, buf_type, alpha_type, chs>(src, dst,
			xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs, fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize);
	} else if (sizeof(_Tp) == 4) { // float
		typedef float value_type; // HResizeCubic/VResizeCubic
		typedef float buf_type;
		typedef float alpha_type;

		resizeGeneric_Cubic<_Tp, value_type, buf_type, alpha_type, chs>(src, dst,
			xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs, fixpt ? (void*)ibeta : (void*)beta, xmin, xmax, ksize);
	} else {
		fprintf(stderr, "not support type\n");
		return -1;
	}

	return 0;
}


int main(){
    cv::Mat mat, matDst1, matDst2;
    mat = cv::imread("img.jpg", 1);//2 | 4
//    std::cout << matSrc;
	for (int i = 0; i < 3; i++) {
		Mat3BGR mat1(mat.rows, mat.cols, mat.data);
		Mat3BGR mat2(mat1);
		Mat3BGR mat3(size[i].height, size[i].width);
		resize_cubic(mat2, mat3, 3);

		cv::Mat mat1_(mat.rows, mat.cols, CV_8UC3, mat.data);
		cv::Mat mat2_;
		mat1_.copyTo(mat2_);
		cv::Mat mat3_(size[i].height, size[i].width, CV_8UC3);
		cv::resize(mat2_, mat3_, cv::Size(size[i].width, size[i].height), 0, 0, 3);

//		assert(mat3.step == mat3_.step);
//		for (int y = 0; y < mat3.rows; y++) {
//			const fbc::uchar* p = mat3.ptr(y);
//			const uchar* p_ = mat3_.ptr(y);
//
//			for (int x = 0; x < mat3.step; x++) {
//				assert(p[x] == p_[x]);
//			}
//		}
        std::cout << mat3;
	}

//	cv::imwrite("cubic_1.jpg", matDst1);
//
//	cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 2);
//	std::cout << matDst1;
//	cv::imwrite("cubic_2.jpg", matDst2);
}

