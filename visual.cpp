#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
//#include "GLSL_helper.h"
#include <math.h>

#include "kmeans.h"

#define X_RANGE 20


int delay = 10;
int time_jump = 50;
bool pause = false;

// some function prototypes
void display(void);
void normalize(float[3]);
void normCrossProd(float[3], float[3], float[3]);

int winWidth;
int winHeight;
float angle = 0.0;
float axis[3];
float trans[3];

bool trackballEnabled = true;
bool trackballMove = false;
bool trackingMouse = false;
bool redrawContinue = false;
bool zoomState = false;
bool shiftState = false;

GLfloat lightXform[4][4] = {
   {1.0, 0.0, 0.0, 0.0},
   {0.0, 1.0, 0.0, 0.0},
   {0.0, 0.0, 1.0, 0.0},
   {0.0, 0.0, 0.0, 1.0}
};

GLfloat objectXform[4][4] = {
   {1.0, 0.0, 0.0, 0.0},
   {0.0, 1.0, 0.0, 0.0},
   {0.0, 0.0, 1.0, 0.0},
   {0.0, 0.0, 0.0, 1.0}
};



GLfloat *trackballXform = (GLfloat *)objectXform;

int step = 0;

// initial viewer position
static GLdouble modelTrans[] = {0.0, 0.0, -5.0};
// initial model angle
static GLfloat theta[] = {0.0, 0.0, 0.0};
static float thetaIncr = 5.0;

// animation transform variables
static GLdouble translate[3] = {-10.0, 0.0, 0.0};

static GLfloat currentTrans[3] = {0.0, 0.0, 0.0};

//---------------------------------------------------------
//   Set up the view

void setUpView() {
   // this code initializes the viewing transform
   glLoadIdentity();

   // moves viewer along coordinate axes
   gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

   // move the view back some relative to viewer[] position
   glTranslatef(0.0f,0.0f, 0.0f);

   // rotates view
   glRotatef(0, 1.0, 0.0, 0.0);
   glRotatef(0, 0.0, 1.0, 0.0);
   glRotatef(0, 0.0, 0.0, 1.0);

   return;
}

//----------------------------------------------------------
//  Set up model transform

void setUpModelTransform() {

   // moves model along coordinate axes
   glTranslatef(modelTrans[0], modelTrans[1], modelTrans[2]);

   // rotates model
   glRotatef(theta[0], 1.0, 0.0, 0.0);
   glRotatef(theta[1], 0.0, 1.0, 0.0);
   glRotatef(theta[2], 0.0, 0.0, 1.0);


}

//----------------------------------------------------------
//  Set up the light

void setUpLight() {
   // set up the light sources for the scene
   // a directional light source from directly behind
   GLfloat lightDir[] = {0.0, 0.0, 5.0, 0.0};
   GLfloat diffuseComp[] = {1.0, 1.0, 1.0, 1.0};

   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   glLightfv(GL_LIGHT0, GL_POSITION, lightDir);
   glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseComp);

   return;
}

//--------------------------------------------------------
//  Set up the objects

GLfloat cc[20][3] = {
{1.0, 0.0, 0.0},
{1.0, 1.0, 0.0},
{1.0, 1.0, 1.0},
{0.0, 0.0, 1.0},
{0.0, 1.0, 1.0},
{0.0, 1.0, 0.2},
{0.3, 0.5, 0.3},
{0.5, 0.3, 0.2},
{0.5, 0.0, 0.2},
{0.2, 0.0, 0.5}
} ;

void drawObjs() {

   // save the transformation state
   glPushMatrix();

   // set the material
   //GLfloat diffuseColor[] = {1.0, 1.0, 0.0, 0.5};
   //glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);
   // orig trans 0 0 1
   //glTranslatef(0, 0, 1);	 // x, y, and z
   //glutSolidCube(2.0);
   
   
   glBegin( GL_POINTS );
   for (int i=0; i<length_data; i++){
      int k = assignments_per_iter[step*length_data+i];		

      glColor3f( cc[k][0],cc[k][1], cc[k][2]);
      //glColor3f( 1.0f * ((float)step/iter), 0.33f, 0.31f );
      glVertex3f(Px[i], Py[i], Pz[i] ); 
		
   }
   glEnd();
   //glFinish();
   
   // recover the transform state
   glPopMatrix();

   return;
}








//-----------------------------------------------------------
//  Callback functions

void reshapeCallback(int w, int h) {
   // from Angel, p.562

   glViewport(0,0,w,h);
   winWidth = w;
   winHeight = h;

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(60.0, 1.0, 0.1, 50.0);

   glMatrixMode(GL_MODELVIEW);
}

float lastPos[3] = {0.0, 0.0, 0.0};
int curx, cury;
int startX, startY;

void trackball_ptov(int x, int y, int width, int height, float v[3]) {
   float d, a;
   // project x, y onto a hemisphere centered within width, height
   v[0] = (2.0*x - width) / width;
   v[1] = (height - 2.0*y) / height;
   d = (float) sqrt(v[0]*v[0] + v[1]*v[1]);
   v[2] = (float) cos((3.14159/2.0) * ((d<1.0)? d : 1.0));
   a = 1.0 / (float) sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
   v[0] *= a;
   v[1] *= a;
   v[2] *= a;
}

void mouseMotion(int x, int y) {
   float curPos[3], dx, dy, dz;

   if (zoomState == false && shiftState == false) {

      trackball_ptov(x, y, winWidth, winHeight, curPos);

      dx = curPos[0] - lastPos[0];
      dy = curPos[1] - lastPos[1];
      dz = curPos[2] - lastPos[2];

      if (dx||dy||dz) {
         angle = 90.0 * sqrt(dx*dx + dy*dy + dz*dz);

         axis[0] = lastPos[1]*curPos[2] - lastPos[2]*curPos[1];
         axis[1] = lastPos[2]*curPos[0] - lastPos[0]*curPos[2];
         axis[2] = lastPos[0]*curPos[1] - lastPos[1]*curPos[0];

         lastPos[0] = curPos[0];
         lastPos[1] = curPos[1];
         lastPos[2] = curPos[2];
      }

   }
   else if (zoomState == true) {
      curPos[1] = y;
      dy = curPos[1] - lastPos[1];

      if (dy) {
         modelTrans[2] += dy * 0.01;
         lastPos[1] = curPos[1];
      }
   }
   else if (shiftState == true) {
      curPos[0] = x; 
      curPos[1] = y;
      dx = curPos[0] - lastPos[0];
      dy = curPos[1] - lastPos[1];

      if (dx) {
         modelTrans[0] += dx * 0.01;
         lastPos[0] = curPos[0];
      }
      if (dy) {
         modelTrans[1] -= dy * 0.01;
         lastPos[1] = curPos[1];
      }
   }
   glutPostRedisplay( );

}

void startMotion(long time, int button, int x, int y) {
   if (!trackballEnabled) return;

   trackingMouse = true;
   redrawContinue = false;
   startX = x; startY = y;
   curx = x; cury = y;
   trackball_ptov(x, y, winWidth, winHeight, lastPos);
   trackballMove = true;
}

void stopMotion(long time, int button, int x, int y) {
   if (!trackballEnabled) return;
   
   trackingMouse = false;

   if (startX != x || startY != y)
      redrawContinue = true;
   else {
      angle = 0.0;
      redrawContinue = false;
      trackballMove = false;
   }
}

void mouseCallback(int button, int state, int x, int y) {

   switch (button) { 
      case GLUT_LEFT_BUTTON:
         trackballXform = (GLfloat *)objectXform;
         break;
      case GLUT_RIGHT_BUTTON:
      case GLUT_MIDDLE_BUTTON:
         trackballXform = (GLfloat *)lightXform;
         break;
   }
   switch (state) {
      case GLUT_DOWN:
         if (button == GLUT_RIGHT_BUTTON) {
            zoomState = true;
            lastPos[1] = y;
         }
         else if (button == GLUT_MIDDLE_BUTTON) {
            shiftState = true;
            lastPos[0] = x;
            lastPos[1] = y;
         }
         else startMotion(0, 1, x, y);
         break;
      case GLUT_UP:
         trackballXform = (GLfloat *)lightXform; // turns off mouse effects
         if (button == GLUT_RIGHT_BUTTON) {
            zoomState = false;
         }
         else if (button == GLUT_MIDDLE_BUTTON) {
            shiftState = false;
         }
         else stopMotion(0, 1, x, y);
         break;
   }
}

void timer(int p){
	
	
	//Update clusters
	if(step < iter-1)
		step++;
			
	if(!pause){
		glutTimerFunc(delay, timer, 1);
	}
	glutPostRedisplay();
	
}
      
void keyCallback(unsigned char key, int x, int y) {
	switch(key){
	case 'd' :
		//Step to next interation
		if(step < iter-1)
			step++;
		break;
	case 'a' :
		//Step to previous interation
		if(step > 0)
			step--;
		break;
	case 'w' :
		//increase speed
		if(delay - time_jump > 0)
			delay -= time_jump;
		break;	
	case 's' :
		//decrease speed
		delay += time_jump;
		break;		
	case 'g' :
		//Run through all interations
		pause = false;
		timer(0);
		break;
	case 'r' :
		//Rest 
		step = 0;
		break;	
	case 'p' :
		//pause 
		pause = true;
		break;	
	}
   glutPostRedisplay();
  
}



//---------------------------------------------------------
//  Main routines

void display (void) {
   // this code executes whenever the window is redrawn (when opened,
   //   moved, resized, etc.
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // set the viewing transform
   setUpView();

   // set up light source
   //setUpLight();

   // start drawing objects
   setUpModelTransform();

   if (trackballMove) {
      glPushMatrix();
      glLoadIdentity();
      glRotatef(angle, axis[0], axis[1], axis[2]);
      glMultMatrixf((GLfloat *) trackballXform);
      glGetFloatv(GL_MODELVIEW_MATRIX, trackballXform);
      glPopMatrix();
   }
   glPushMatrix();
   glMultMatrixf((GLfloat *) objectXform);

   drawObjs();
   //drawBox();
   //drawTorus();
   //drawCone(); 
   //drawSphere();
 
   glPopMatrix();

   glutSwapBuffers();

}

// create a double buffered 500x500 pixel color window
extern "C" int drawEverything(void) {

    int myargc = 1;
    char *myargv[7]= { "visual" }; 

    glutInit(&myargc, myargv);
      
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(800, 800);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("k-means clustering");
    
    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 4.0 );
    
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(display);
    glutReshapeFunc(reshapeCallback);
    glutKeyboardFunc(keyCallback);
    glutMouseFunc(mouseCallback);
    glutMotionFunc(mouseMotion);
	glutMainLoop();
	return 0;
}
