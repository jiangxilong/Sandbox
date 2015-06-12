#include <string.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

#include <unistd.h>

#include "libfreenect.hpp"
#include <libfreenect.h>
#include <pthread.h>


#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <opencv/cxcore.h>
#include "opencv2/videostab/inpainting.hpp"

using namespace cv;
using namespace std;

#define FREENECTOPENCV_WINDOW_D "Depthimage"
#define FREENECTOPENCV_WINDOW_N "Normalimage"
#define FREENECTOPENCV_RGB_DEPTH 3
#define FREENECTOPENCV_DEPTH_DEPTH 1
#define FREENECTOPENCV_RGB_WIDTH 640
#define FREENECTOPENCV_RGB_HEIGHT 480
#define FREENECTOPENCV_DEPTH_WIDTH 640
#define FREENECTOPENCV_DEPTH_HEIGHT 480
#define FREENECT_FRAME_W   640
#define FREENECT_FRAME_H   480
#define FREENECT_FRAME_PIX   (FREENECT_FRAME_H*FREENECT_FRAME_W)
#define FREENECT_VIDEO_RGB_SIZE   (FREENECT_FRAME_PIX*3)


//blub

IplImage* depthimg = 0;
IplImage* rgbimg = 0;
IplImage* tempimg = 0;
pthread_mutex_t mutex_depth = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_rgb = PTHREAD_MUTEX_INITIALIZER;
pthread_t cv_thread;


// callback for depthimage, called by libfreenect
void depth_cb(freenect_device *dev, void *depth, uint32_t timestamp)

{
        cv::Mat depth8;
        cv::Mat mydepth = cv::Mat( FREENECTOPENCV_DEPTH_WIDTH,FREENECTOPENCV_DEPTH_HEIGHT, CV_16UC1, depth);

        mydepth.convertTo(depth8, CV_8UC1, 1.0/4.0);
        pthread_mutex_lock( &mutex_depth );
        memcpy(depthimg->imageData, depth8.data, 640*480);
        // unlock mutex
        pthread_mutex_unlock( &mutex_depth );

}



// callback for rgbimage, called by libfreenect

void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp) {
    // lock mutex for opencv rgb image
    pthread_mutex_lock( &mutex_rgb );
    memcpy(rgbimg->imageData, rgb, FREENECT_VIDEO_RGB_SIZE);
    // unlock mutex
    pthread_mutex_unlock( &mutex_rgb );
}


/*
 * thread for displaying the opencv content
 */
void *cv_threadfunc (void *ptr) {
        cvNamedWindow( FREENECTOPENCV_WINDOW_D, CV_WINDOW_AUTOSIZE );
        cvNamedWindow( FREENECTOPENCV_WINDOW_N, CV_WINDOW_AUTOSIZE );
        depthimg = cvCreateImage(cvSize(FREENECTOPENCV_DEPTH_WIDTH, FREENECTOPENCV_DEPTH_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_DEPTH_DEPTH);
        rgbimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);
        tempimg = cvCreateImage(cvSize(FREENECTOPENCV_RGB_WIDTH, FREENECTOPENCV_RGB_HEIGHT), IPL_DEPTH_8U, FREENECTOPENCV_RGB_DEPTH);

        //use image polling
        while (1) {
                //lock mutex for depth image
                pthread_mutex_lock( &mutex_depth );
                //show image to window
                cvCvtColor(depthimg,tempimg,CV_GRAY2BGR);
                cvCvtColor(tempimg,tempimg,CV_HSV2BGR);
                cvShowImage(FREENECTOPENCV_WINDOW_D,depthimg);
                //unlock mutex for depth image
                pthread_mutex_unlock( &mutex_depth );

                //lock mutex for rgb image
                pthread_mutex_lock( &mutex_rgb );
                //show image to window
                cvCvtColor(rgbimg,tempimg,CV_BGR2RGB);
                cvShowImage(FREENECTOPENCV_WINDOW_N, tempimg);
                //unlock mutex
                pthread_mutex_unlock( &mutex_rgb );

                //wait for quit key
                if( cvWaitKey( 15 )==27 ){
                    break;
                }

        }
        pthread_exit(NULL);

}


class myMutex {
	public:
		myMutex() {
			pthread_mutex_init( &m_mutex, NULL );
		}
		void lock() {
			pthread_mutex_lock( &m_mutex );
		}
		void unlock() {
			pthread_mutex_unlock( &m_mutex );
		}
	private:
		pthread_mutex_t m_mutex;
};


class MyFreenectDevice : public Freenect::FreenectDevice {
	public:
		MyFreenectDevice(freenect_context *_ctx, int _index)
	 		: Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),
			m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false),
			m_new_depth_frame(false), depthMat(Size(640,480),CV_16UC1),
			rgbMat(Size(640,480), CV_8UC3, Scalar(0)),
			ownMat(Size(640,480),CV_8UC3,Scalar(0)) {

			for( unsigned int i = 0 ; i < 2048 ; i++) {
				float v = i/2048.0;
				v = std::pow(v, 3)* 6;
				m_gamma[i] = v*6*256;
			}
		}

		// Do not call directly even in child
		void VideoCallback(void* _rgb, uint32_t timestamp) {
			//std::cout << "RGB callback" << std::endl;
			m_rgb_mutex.lock();
			uint8_t* rgb = static_cast<uint8_t*>(_rgb);
			rgbMat.data = rgb;
			m_new_rgb_frame = true;
			m_rgb_mutex.unlock();
		};

		// Do not call directly even in child
		void DepthCallback(void* _depth, uint32_t timestamp) {

			//reduziert die Framerate
			//usleep(250*1000);

			//std::cout << "Depth callback" << std::endl;
			m_depth_mutex.lock();
			uint16_t* depth = static_cast<uint16_t*>(_depth);
			depthMat.data = (uchar*) depth;
			m_new_depth_frame = true;
			m_depth_mutex.unlock();
		}

		bool getVideo(Mat& output) {
			m_rgb_mutex.lock();
			if(m_new_rgb_frame) {
				cv::cvtColor(rgbMat, output, CV_RGB2BGR);
				m_new_rgb_frame = false;
				m_rgb_mutex.unlock();
				return true;
			} else {
				m_rgb_mutex.unlock();
				return false;
			}
		}

		bool getDepth(Mat& output) {
				m_depth_mutex.lock();
				if(m_new_depth_frame) {
					depthMat.copyTo(output);
					m_new_depth_frame = false;
					m_depth_mutex.unlock();
					return true;
				} else {
					m_depth_mutex.unlock();
					return false;
				}
			}
	private:
		std::vector<uint8_t> m_buffer_depth;
		std::vector<uint8_t> m_buffer_rgb;
		std::vector<uint16_t> m_gamma;
        bool m_new_rgb_frame;
		bool m_new_depth_frame;
		Mat depthMat;
		Mat rgbMat;
		Mat ownMat;
		myMutex m_rgb_mutex;
		myMutex m_depth_mutex;
};


int main(int argc, char **argv) {

        bool die(false);

        Mat depthMat(Size(640,480), CV_16UC1);
        Mat depthf(Size(640,480), CV_8UC1);
        Mat tempf(Size(640,480), CV_8UC3);
        Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
        Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));

        Freenect::Freenect freenect;
        MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

        device.startVideo();
        device.startDepth();

        while (!die) {
            device.getVideo(rgbMat);
            device.getDepth(depthMat);

            //cvSetCaptureProperty(depthMat, CV_CAP_PROP_FPS, 15);

            //interpolation & inpainting
            Mat _tmp,_tmp1; //minimum observed value is ~440. so shift a bit
            Mat(depthMat - 200.0).convertTo(_tmp1,CV_64FC1);

            Point minLoc; double minval,maxval;
            minMaxLoc(_tmp1, &minval, &maxval, NULL, NULL);
            //cout << maxval << endl;
            _tmp1.convertTo(depthf, CV_8UC1, 0.28, 0);  //linear interpolation

            //use a smaller version of the image
            //Mat small_depthf; resize(depthf,small_depthf,Size(),0.2,0.2);
            //inpaint only the "unknown" pixels
            cv::inpaint(depthf,(depthf == 255),_tmp1,5.0,INPAINT_TELEA);

            resize(_tmp1, _tmp, depthf.size());
            _tmp.copyTo(depthf, (depthf == 255));  //add the original signal back over the inpaint

            //add color to grey
            cv::cvtColor(depthf, tempf, CV_GRAY2BGR);
            cv::cvtColor(tempf, tempf, CV_HSV2BGR);

            cv::imshow("depth",tempf);
            char k = cvWaitKey(5);
            if( k == 27 ){
                break;
            }

        }

        device.stopVideo();
        device.stopDepth();
        return 0;

}
