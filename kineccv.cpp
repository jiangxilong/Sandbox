#include <string.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>

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


int main(int argc, char **argv)
{

        freenect_context *f_ctx;
        freenect_device *f_dev;


        int res = 0;
        int die = 0;
        printf("Kinect camera test\n");

        if (freenect_init(&f_ctx, NULL) < 0) {
                        printf("freenect_init() failed\n");
                        return 1;
                }

        if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
                printf("Could not open device\n");
                return 1;
        }

        freenect_set_depth_callback(f_dev, depth_cb);
        freenect_set_video_callback(f_dev, rgb_cb);
        //freenect_video_format requested_format(FREENECT_VIDEO_RGB);

        //create opencv display thread
        res = pthread_create(&cv_thread, NULL, cv_threadfunc, (void*) depthimg);
        if (res) {
                printf("pthread_create failed\n");
                return 1;
        }

        printf("init done\n");



        freenect_start_depth(f_dev);
        freenect_start_video(f_dev);

        while(!die && freenect_process_events(f_ctx) >= 0 );




        Mat depthMat(Size(640,480),CV_16UC1);
        Mat depthf  (Size(640,480),CV_8UC1);
        Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
        Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));

        Freenect::Freenect freenect;
        MyFreenectDevice& f_dev = freenect.createDevice<MyFreenectDevice>(0);

        freenect_start_video(f_dev);
        freenect_start_depth(f_dev);

        while (!die) {
            f_dev.getVideo(rgbMat);
            f_dev.getDepth(depthMat);
            depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
            cv::imshow("depth",depthf);
            char k = cvWaitKey(5);
            if( k == 27 ){
                break;
            }
        }

        freenect_stop_video(f_dev);
        freenect_stop_depth(f_dev);

        //interpolation & inpainting

        Mat _tmp,_tmp1; //minimum observed value is ~440. so shift a bit
        Mat(depthMat - 400.0).convertTo(_tmp1,CV_64FC1);

        Point minLoc; double minval,maxval;
        minMaxLoc(_tmp1, &minval, &maxval, NULL, NULL);
        _tmp1.convertTo(depthf, CV_8UC1, 255.0/maxval);  //linear interpolation

        //use a smaller version of the image
        Mat small_depthf; resize(depthf,small_depthf,Size(),0.2,0.2);
        //inpaint only the "unknown" pixels
        cv::inpaint(small_depthf,(small_depthf == 255),_tmp1,5.0,INPAINT_TELEA);

        resize(_tmp1, _tmp, depthf.size());
        _tmp.copyTo(depthf, (depthf == 255));  //add the original signal back over the inpaint




}
