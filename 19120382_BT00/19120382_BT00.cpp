#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#define M_PI   3.14159265358979323846264338327950288
#define STRONG 255
#define WEAK 0
using namespace std;
using namespace cv;
//-----------------------------------------------------------------------
// Thanh vien:
// 1/ Nguyen Hai Ha - 19120215
// 2/ Vo Tien Thinh - 19120382
////////////////////////////////////////////////////////////////////


//-----------------------------------------------------------------
//                      HUONG DAN CHAY CHUONG TRINH
//  Dung command line de chay chuong trinh
//	Chuong trinh duoc build duoi dang release nhu de bai yeu cau
//  Command line cua cac chuc nang:
//		+Mo anh: <ten chuong trinh> <duong dan tap tin anh>
//		+Phat hien bien canh su dung Prewitt: <ten chuong trinh> <duong dan tap tin anh> detecByPrewitt
//		+Phat hien bien canh su dung Canny: <ten chuong trinh> <duong dan tap tin anh> detecByCanny <Low Threshold Ratio> <High Threshold Ratio>
//		*Low Threshold Ratio la ti le chan duoi khi so voi High Threshold
//		*High Threshold Ratio la ti le chan tren khi so voi gia tri pixel lon nhat trong anh (anh da gray scale)
//		*Trong thuc nghiem, Low Threshold Ratio va High Threshold Ratio cho ket qua kha quan lan luot la: 0.06 0.11 
/////////////////////////////////////////////////////////////////////////////////////


/////////////////////////// Ham cho thuat Prewitt//////////////////////////////////
Mat xGradientOfPrewitt(Mat image) ////// Tinh anh gradient theo huong x
{
	Mat GxKernelPrewitt = (Mat_<char>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1); ////kernel x Prewitt
	Mat GxPrewitt;
	filter2D(image, GxPrewitt, CV_32F, GxKernelPrewitt); /////////////// kernel Prewitt x * source image
	imshow("Gradient Image x", GxPrewitt);
	return GxPrewitt;
}
Mat yGradientOfPrewitt(Mat image)
{
	Mat GyKernelPrewitt = (Mat_<char>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1); ////kernel y Prewitt
	Mat GyPrewitt;
	filter2D(image, GyPrewitt, CV_32F, GyKernelPrewitt); /////////////// kernel y Prewitt * source image
	imshow("Gradient Image y", GyPrewitt);
	return GyPrewitt;
}

int detectByPrewitt(Mat src, Mat &dst)
{
	Mat Gx, Gy,squGx,squGy, sum;
	Gx = xGradientOfPrewitt(src);
	Gy = yGradientOfPrewitt(src);
	pow(Gx, 2, squGx); 
	pow(Gy, 2, squGy);  
	sum = squGx + squGy;
	sqrt(sum, dst); ///////// dst= can bac hai ( Gx^2 + Gy^2)
	dst = dst * (1.0 / 255); //////////////// imshow yeu cau range [0,1] cho anh floating point nen can chia 255 de dua ve khoang do
	return 1;
}
//////////////////////////////////////////////

/////////////////////Ham cho thuat Canny//////////////////////////
Mat xGradientOfSobel(Mat image) ////////Tinh anh gradient theo huong x voi Sobel kernel
{
	Mat GxKernelSobel = (Mat_<char>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat GxSobel;
	filter2D(image, GxSobel, CV_32F, GxKernelSobel);
	return GxSobel;
}
Mat yGradientOfSobel(Mat image) ////////Tinh anh gradient theo huong y voi Sobel kernel
{
	Mat GyKernelSobel = (Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat GySobel;
	filter2D(image, GySobel, CV_32F, GyKernelSobel);
	return GySobel;
}

bool isBetween(float angle, float a, float b, float c, float d) { //////////////////
	if ((angle >= a && angle <= b) || (angle >= c && angle <= d)) 
	{
		return true;
	}
	else 
	{
		return false;
	}
}

float getOrientation(float angle)
{
	if (isBetween(angle, -22.5, 22.5, -180, -157.5) || isBetween(angle, 157.5, 180, -22.5, 0))
		return 0;
	else if (isBetween(angle, 22.5, 67.5, -157.5, -112.5))
		return 45;
	else if (isBetween(angle, 67.5, 112.5, -112.5, -67.5))
		return 90;
	else if (isBetween(angle, 112.5, 157.5, -67.5, -22.5))
		return 135;
	else
		return -1;
}

Mat nonMaxSuppression(Mat image, Mat &theta)
{
	Mat angle, outputNonMaxSup;
	float value, q, r;
	q = 255;
	r = 255;
	
	angle = theta * 180.0/ M_PI; // Tao ra ma tran goc (angle matrix)/////////
	outputNonMaxSup = image.clone();
	//////// Tim huong (edge direction) cua moi pixel dua vao ma tran goc//////
	for (int i = 0; i < theta.rows; i++)  
	{
		for (int j = 0; j < theta.cols; j++)
		{
			float value = angle.at<float>(i, j);
			if (value > 180)
			{
				value = value - 360;
			}
			angle.at<float>(i, j) = getOrientation(value);
		}	
	}
	//////////////// Kiem tra moi pixel co gia tri lon hon cac pixel cung huong goc (same direction) khong
	//////////////// Neu co thi giu nguyen gia tri
	//////////////// Neu khong thi doi pixel do sang mau den 
	for (int i = 0; i < theta.rows; i++)
	{
		for (int j = 0; j < theta.cols; j++)
		{
			if (angle.at<float>(i, j) == 0)
			{
				q = image.at<float>(i, j + 1);
				r = image.at<float>(i, j - 1);
			}
			else if (angle.at<float>(i, j) == 45)
			{
				q = image.at<float>(i+1, j - 1);
				r = image.at<float>(i-1, j + 1);
			}
			else if (angle.at<float>(i, j) == 90)
			{
				q = image.at<float>(i+1, j);
				r = image.at<float>(i-1, j);
			}
			else if (angle.at<float>(i, j) == 135)
			{
				q = image.at<float>(i-1, j - 1);
				r = image.at<float>(i+1, j + 1);
			}

			if (image.at<float>(i, j) >= q && image.at<float>(i, j) >= r)
				outputNonMaxSup.at<float>(i, j) = image.at<float>(i, j);
			else
				outputNonMaxSup.at<float>(i, j) = 0.0;
		}
	}
	return outputNonMaxSup;
}

////////// Tao mot ma tran luu ket qua so sanh gia tri moi pixel voi Threshold
////////// Gia tri pixel > High threshold. Luu ket qua so sanh la 1
////////// Gia tri pixel < Low threshold. Luu ket qua so sanh la -1
///////// Low threshold <= gia tri pixel <= High threshold. Luu ket qua so sanh la 0
Mat doubleThreshold(Mat img, float lowThresholdRatio, float highThresholdRatio)
{
	Mat outputThreshold;
	double min, max;
	minMaxIdx(img, &min, &max);
	float highThreshold = max * highThresholdRatio;
	float lowThreshold = lowThresholdRatio * highThreshold;
	outputThreshold = img.clone();
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<float>(i, j) > highThreshold)
				outputThreshold.at<float>(i, j) = 1;
			else if (img.at<float>(i, j) < lowThreshold)
				outputThreshold.at<float>(i, j) = -1;
			else outputThreshold.at<float>(i, j) = 0;
		}
	}
	return outputThreshold;
}


/////// Kiem tra 8 pixel xung quanh pixel dang xet. Neu co mot pixel neighbor la strong pixel thi pixel dang xet se duoc xem la strong pixel
/////// Nguoc lai se duoc xem la weak pixel.
bool checkNeighbor(Mat type, int i, int j)
{

	if (type.at<float>(i + 1, j - 1) == 1 || type.at<float>(i + 1, j) == 1 || type.at<float>(i + 1, j + 1) == 1 || type.at<float>(i, j - 1) == 1
		|| type.at<float>(i, j + 1) == 1 || type.at<float>(i - 1, j - 1) == 1 || type.at<float>(i - 1, j) == 1 || type.at<float>(i - 1, j + 1) == 1)
	{
		return true;
	}
	else
		return false;

}

//////// Su dung bang ket qua so sanh threshold de ve duong bien canh
//////// Pixel co ket qua so sanh bang 1 duoc xem la strong pixel. Va duoc gan mau trang.
//////// Pixel co ket qua so sanh bang -1 duoc xem la weak pixel. Va duoc gan mau den.
//////// Pixel co ket qua so sanh bang 0 duoc xet them buoc checkNeighbor da trinh bay o tren.
Mat hysteresis(Mat type)
{
	Mat outputHysteresis = type.clone();
	for (int i = 1; i < type.rows-1; i++)
	{
		for (int j = 1; j < type.cols-1; j++)
		{
			if (type.at<float>(i, j) == -1)
				outputHysteresis.at<float>(i, j) = WEAK;
			else if(type.at<float>(i, j) == 1)
				outputHysteresis.at<float>(i, j) = STRONG;
			else
			{
				if (checkNeighbor(type, i, j))
				{
					outputHysteresis.at<float>(i, j) = STRONG;
				}
				else
					outputHysteresis.at<float>(i, j) = WEAK;
			}
		}
	}
	return outputHysteresis;

}

int detecByCanny(Mat src, Mat& dst, float lowThresholdRatio, float highThresholdRatio)
{
	Mat Gx, Gy, sqGx, sqGy, gSobel,theta, gNonMaxSup, gThreshold;
	double min,max;
	//////// Su dung thuat Sobel de tim magnitude cua anh //////////////
	Gx = xGradientOfSobel(src);
	Gy = yGradientOfSobel(src);
	pow(Gx, 2, sqGx);
	pow(Gy, 2, sqGy);
	gSobel = sqGx + sqGy;
	sqrt(gSobel, gSobel);
	minMaxIdx(gSobel, &min, &max);
	gSobel = gSobel / max * 255; 
	////////////////////////////////////
	phase(Gx,Gy,theta); ////////////// Tinh theta = arctan(Gy/Gx)
	gNonMaxSup =nonMaxSuppression(gSobel,theta);
	gThreshold = doubleThreshold(gNonMaxSup, lowThresholdRatio, highThresholdRatio);
	dst = hysteresis(gThreshold);
	return 1;
}

int main(int argc, char** argv)
{
	if (argc < 2) //Hien ten chuong trinh
	{
		cout << "Chuong trinh mo va hien thi anh" << endl;
		return -1;
	}
	else if (argc == 3||argc == 5)
	{
		Mat image;
		image = imread(argv[1], IMREAD_COLOR);
		if (!image.data)
		{
			cout << "Khong the mo anh" << std::endl;
			return -1;
		}
		else if (argc == 3 && std::string(argv[2]) == "detecByPrewitt") ////////////// xac dinh bien canh bang Prewitt//////////
		{
			Mat imageGray, imageBlur;
			Mat detectedEdgeImage;
			cvtColor(image, imageGray, COLOR_BGR2GRAY); ////////Grayscale anh
			GaussianBlur(imageGray, imageBlur, Size(3, 3), 0); /////////Lam mo anh
			detectByPrewitt(imageBlur, detectedEdgeImage);
			imshow("Source image", image);
			imshow("Detected by Prewitt image", detectedEdgeImage);
			waitKey(0);
			return 0;
		}
		else if (argc == 5 && std::string(argv[2]) == "detecByCanny") ////////////// xac dinh bien canh bang Canny//////////
		{
			Mat imageGray, imageBlur;
			Mat detectedEdgeImage;
			float lowThresholdRatio = atof(argv[3]);
			float highThresholdRatio = atof(argv[4]);
			cvtColor(image, imageGray, COLOR_BGR2GRAY); ////////Grayscale anh
			GaussianBlur(imageGray, imageBlur, Size(3, 3), 0);/////////Lam mo anh
			detecByCanny(imageBlur, detectedEdgeImage, lowThresholdRatio, highThresholdRatio);
			imshow("Source image", image);
			imshow("Detected by Canny image", detectedEdgeImage);
			waitKey(0);
			return 0;
		}
	}
	else
	{
		return 0;
	}
}
