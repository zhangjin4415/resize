#include <opencv2/opencv.hpp>
//#include <iostream>
// using namespace std;

//template<typename _Tp> class Size_;
template<typename _Tp> class Rect_;
template<typename _Tp, int cn> class Vec;
template<typename _Tp> class Scalar_;
template<typename _Tp> class Point_;
//////////////////////////////// Scalar_ ///////////////////////////////
// Template class for a 4-element vector derived from Vec
template<typename _Tp> class Scalar_ : public Vec<_Tp, 4>
{
public:
	//! various constructors
	Scalar_();
	Scalar_(_Tp v0, _Tp v1, _Tp v2 = 0, _Tp v3 = 0);
	Scalar_(_Tp v0);

	template<typename _Tp2, int cn>
	Scalar_(const Vec<_Tp2, cn>& v);

	//! returns a scalar with all elements set to v0
	static Scalar_<_Tp> all(_Tp v0);

	//! conversion to another data type
	template<typename T2> operator Scalar_<T2>() const;

	//! per-element product
	Scalar_<_Tp> mul(const Scalar_<_Tp>& a, double scale = 1) const;

	// returns (v0, -v1, -v2, -v3)
	Scalar_<_Tp> conj() const;

	// returns true iff v1 == v2 == v3 == 0
	bool isReal() const;
};

typedef Scalar_<double> Scalar;


//////////////////////////////// Rect_ ////////////////////////////////
// Template class for 2D rectangles
template<typename _Tp> class Rect_
{
public:
	typedef _Tp value_type;

	//! various constructors
	Rect_();
	Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
	Rect_(const Rect_& r);
	Rect_(const Point_<_Tp>& org, const Size_<_Tp>& sz);
	Rect_(const Point_<_Tp>& pt1, const Point_<_Tp>& pt2);

	Rect_& operator = (const Rect_& r);
	//! the top-left corner
	Point_<_Tp> tl() const;
	//! the bottom-right corner
	Point_<_Tp> br() const;

	//! size (width, height) of the rectangle
	Size_<_Tp> size() const;
	//! area (width*height) of the rectangle
	_Tp area() const;

	//! conversion to another data type
	template<typename _Tp2> operator Rect_<_Tp2>() const;

	//! checks whether the rectangle contains the point
	bool contains(const Point_<_Tp>& pt) const;

	_Tp x, y, width, height; //< the top-left corner, as well as width and height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;


////////////////////////////// Small Matrix ///////////////////////////
// Template class for small matrices whose type and size are known at compilation time
template<typename _Tp, int m, int n> class Matx {
public:
	enum {
		rows = m,
		cols = n,
		channels = rows*cols,
		shortdim = (m < n ? m : n)
	};

	typedef _Tp value_type;
	typedef Matx<_Tp, m, n> mat_type;
	typedef Matx<_Tp, shortdim, 1> diag_type;

	//! default constructor
	Matx();

	Matx(_Tp v0); //!< 1x1 matrix
	Matx(_Tp v0, _Tp v1); //!< 1x2 or 2x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2); //!< 1x3 or 3x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 1x4, 2x2 or 4x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 1x5 or 5x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 1x6, 2x3, 3x2 or 6x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 1x7 or 7x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 1x8, 2x4, 4x2 or 8x1 matrix
	Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 1x9, 3x3 or 9x1 matrix
	explicit Matx(const _Tp* vals); //!< initialize from a plain array

	static Matx all(_Tp alpha);
	static Matx zeros();
	static Matx ones();
	static Matx eye();
	static Matx diag(const diag_type& d);

	//! dot product computed with the default precision
	_Tp dot(const Matx<_Tp, m, n>& v) const;

	//! dot product computed in double-precision arithmetics
	double ddot(const Matx<_Tp, m, n>& v) const;

	//! conversion to another data type
	template<typename T2> operator Matx<T2, m, n>() const;

	//! change the matrix shape
	template<int m1, int n1> Matx<_Tp, m1, n1> reshape() const;

	//! extract part of the matrix
	template<int m1, int n1> Matx<_Tp, m1, n1> get_minor(int i, int j) const;

	//! extract the matrix row
	Matx<_Tp, 1, n> row(int i) const;

	//! extract the matrix column
	Matx<_Tp, m, 1> col(int i) const;

	//! extract the matrix diagonal
	diag_type diag() const;

	//! element access
	const _Tp& operator ()(int i, int j) const;
	_Tp& operator ()(int i, int j);

	//! 1D element access
	const _Tp& operator ()(int i) const;
	_Tp& operator ()(int i);

	_Tp val[m*n]; //< matrix elements
};


///////////////////////////////////////// Vec ///////////////////////////////////
// Template class for short numerical vectors, a partial case of Matx
template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1> {
public:
	typedef _Tp value_type;
	enum {
		channels = cn
	};

	//! default constructor
	Vec();

	Vec(_Tp v0); //!< 1-element vector constructor
	Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2); //!< 3-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 4-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 5-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 6-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 7-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 8-element vector constructor
	Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 9-element vector constructor
	explicit Vec(const _Tp* values);

	Vec(const Vec<_Tp, cn>& v);

	static Vec all(_Tp alpha);

	//! per-element multiplication
	Vec mul(const Vec<_Tp, cn>& v) const;

	//! conversion to another data type
	template<typename T2> operator Vec<T2, cn>() const;

	/*! element access */
	const _Tp& operator [](int i) const;
	_Tp& operator[](int i);
	const _Tp& operator ()(int i) const;
	_Tp& operator ()(int i);
};

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



//////////////////////////////// Point_ ////////////////////////////////
// Template class for 2D points specified by its coordinates `x` and `y`
template<typename _Tp> class Point_
{
public:
	typedef _Tp value_type;

	// various constructors
	Point_();
	Point_(_Tp _x, _Tp _y);
	Point_(const Point_& pt);
	Point_(const Size_<_Tp>& sz);
	Point_(const Vec<_Tp, 2>& v);

	Point_& operator = (const Point_& pt);
	//! conversion to another data type
	template<typename _Tp2> operator Point_<_Tp2>() const;

	//! conversion to the old-style C structures
	operator Vec<_Tp, 2>() const;

	//! dot product
	_Tp dot(const Point_& pt) const;
	//! dot product computed in double-precision arithmetics
	double ddot(const Point_& pt) const;
	//! cross-product
	double cross(const Point_& pt) const;
	//! checks whether the point is inside the specified rectangle
	bool inside(const Rect_<_Tp>& r) const;

	_Tp x, y; //< the point coordinates
};

typedef Point_<int> Point2i;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;



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

typedef unsigned char uchar;
int main(){
    cv::Mat mat, matDst1, matDst2;
    mat = cv::imread("img.jpg", 1);//2 | 4
    int width = 623, height = 711;
	cv::cvtColor(mat, mat, CV_BGR2GRAY);
	mat.convertTo(mat, CV_32FC1);
//    std::cout << matSrc;
	Mat_<float, 1> mat1(mat.rows, mat.cols, mat.data);
    Mat_<float, 1> mat2(mat1);
    Mat_<float, 1> mat3(height, width);
    resize_cubic(mat2, mat3);

    cv::Mat mat1_(mat.rows, mat.cols, CV_32FC1, mat.data);
    cv::Mat mat2_;
    mat1_.copyTo(mat2_);
    cv::Mat mat3_(height, width, CV_32FC1);
    cv::resize(mat2_, mat3_, cv::Size(width, height), 0, 0, 2);

    assert(mat3.step == mat3_.step);
    for (int y = 0; y < mat3.rows; y++) {
        const uchar* p = mat3.ptr(y);
        const uchar* p_ = mat3_.ptr(y);

        for (int x = 0; x < mat3.step; x++) {
            assert(p[x] == p_[x]);
        }
    }
    // std::cout << mat3;
//	cv::imwrite("cubic_1.jpg", matDst1);
//
//	cv::resize(matSrc, matDst2, matDst1.size(), 0, 0, 2);
//	std::cout << matDst1;
//	cv::imwrite("cubic_2.jpg", matDst2);
}

