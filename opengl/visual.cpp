#ifdef __APPLE__
#include "GLUT/glut.h"
#include <OPENGL/gl.h>
#endif
#ifdef __unix__
#include <GL/glut.h>
#endif

#include <stdlib.h>
#include <stdio.h>
//#include "GLSL_helper.h"
#include <math.h>
#include "kmeans.h"

#define X_RANGE 50

// some function prototypes
void display(void);
void normalize(float[3]);
void normCrossProd(float[3], float[3], float[3]);

int winWidth, winHeight;
float angle = 0.0, axis[3], trans[3];
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

float* xnorm;
float* ynorm;
float* znorm;

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
   
   glColor3f( 0.95f, 0.0f, 0.031f );
   
   glBegin( GL_POINTS );
   for (int i=0; i<10; i++){
      
		glVertex3f(xnorm[i], ynorm[i], znorm[i] ); 
		
   }
   glEnd();
   //glFinish();
   
   // recover the transform state
   glPopMatrix();

   return;
}


void drawTorus() {

   // prepare to draw lower torus
   glPushMatrix();

   // set the material
   GLfloat diffuseColor[] = {1.0, 0.0, 1.0, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);

   /* draw the lower Torus - parameters are inner radius, outer radius, 
   num sides, num rings */
   glutSolidTorus(.3, 2, 30, 30);

   glPopMatrix();
   

   // prepare to draw middle torus
   glPushMatrix();

   // set the material
   GLfloat diffuseColor2[] = {1, 1, 1, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor2);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);

   // Rotate about the bottom scoop
   glTranslatef(0, 0, -1.2);
   glRotatef(7, 1, 0, 0);
   glTranslatef(0, 0, -0.5);
   
   glutSolidTorus(.25, 1.8, 30, 30);

   glPopMatrix();

   // prepare to draw upper torus
   glPushMatrix();

   // set the material
   GLfloat diffuseColor3[] = {0.2, 1, 0.2, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor3);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);

   // Rotate about the bottom scoop
   glTranslatef(0, .2, -2.6);
   glRotatef(-1, 1, 0, 0);
   glTranslatef(0, 0, -0.5);
   
   
   glutSolidTorus(.22, 1.73, 30, 30);

   glPopMatrix();

   return;
}

void drawSphere() {
   
   // prepare to draw lower sphere
   glPushMatrix();

   // set the material
   GLfloat diffuseColor[] = {1.0, 0.0, 1.0, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);

   
   glTranslatef(0, 0, -0.5);	 // x, y, and z

   // draw the sphere - parameters are radius, number slices, and number stacks
   glutSolidSphere(2, 30, 30);

   glPopMatrix();


   // prepare to draw middle sphere
   glPushMatrix();

   GLfloat diffuseColor2[] = {1, 1, 1, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor2);

   // rotate around bottom sphere
   glTranslatef(0, 0, -1.6);
   glRotatef(20, 1, 0, 0);
   glTranslatef(0, 0, -0.5);


   glutSolidSphere(1.8, 30, 30);

   // recover the transform state
   glPopMatrix();


   // prepare to draw upper sphere
   glPushMatrix();

   GLfloat diffuseColor3[] = {0.2, 1, 0.2, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor3);

   // rotate around bottom sphere
   glTranslatef(0, 0, -3.1);
   glRotatef(12, 1, 0, 0);
   glTranslatef(0, 0, -0.5);


   glutSolidSphere(1.7, 30, 30);

   // recover the transform state
   glPopMatrix();

   return;
}


void drawCone() {

   // save the transformation state
   glPushMatrix();

   // set the material
   GLfloat diffuseColor[] = {0.9, 0.7, 0.05, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);
   // Adjust the translate and rotation to make the cone the right
   //    shape and in the right place
   glTranslatef(0, 0, 0);	 // x, y, and z
   glRotatef(0, 0, 0, 0);  // angle and axis (x, y, z components)
   // draw the cone - parameters are bottom radius, height, and number
   // of slices horizontally and radially
   glutSolidCone(2.0, 5.0, 30, 30);

   // recover the transform state
   glPopMatrix();

   return;
}


void drawBox() {

   int i;

   // set the material
   GLfloat diffuseColor[] = {0.1, 0.1, 0.1, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor);

   // locate it in the scene
   glMatrixMode(GL_MODELVIEW);

   // Make sprinkls/choclate chips
   for(i = 0; i < 20; i++){

      //Place on top of scoop surface and rotate down/over
      //glTranslatef(0, 0, -2);

      glPushMatrix();
      glTranslatef(0, 0, -3.5);
      glRotatef(i*10, (i%3), (i%2), 1);
      //glRotatef(i*-15, (i%5), (i%7), 1);
      glTranslatef(0, 0, -1.7);
   
      //rotate chip randomly-ish
      glRotatef(20*i, 0 , 0, 1);
      glRotatef(10*i, 1 , 1, 0);

      //Make flatter
      glScalef(5,3,1);

      // draw the box - parameter is side length
      glutSolidCube(.1);

      glPopMatrix();

   }

   // set the material
   GLfloat diffuseColor2[] = {1, 0.1, 0.1, 1.0};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuseColor2);

   // Make sprinkls/choclate chips
   for(i = 0; i < 36; i++){

      //Place on top of scoop surface and rotate down/over
      //glTranslatef(0, 0, -2);

      glPushMatrix();
      glTranslatef(0.2, 0.2, -3.3);
      glRotatef(i*10, (i%2), (i%3), 1);
      glTranslatef(0.5, 0.5, -1.7);
   
      //rotate chip randomly-ish
      glRotatef(20*i, 0 , 0, 1);
      glRotatef(10*i, 1 , 1, 0);

      //Make flatter
      glScalef(5,3,1);

      // draw the box - parameter is side length
      glutSolidCube(.1);

      glPopMatrix();

   }

   // draw the box - parameter is side length
   glutSolidCube(1);

   

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
      
void keyCallback(unsigned char key, int x, int y) {

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
int mainDraw() {
	glutInit(NULL, NULL);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Mouse motion example");
    printf("Interaction directions:\n\nLeft mouse button: hold down and drag to change orientation\n");
    printf("Middle mouse button: hold down and drag to shift horizontally or vertically\n");
    printf("Right mouse button: hold down and drag vertically to move in and out of the screen\n");
    
    glEnable( GL_POINT_SMOOTH );
    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glPointSize( 6.0 );
    
    //Normalize data TODO DATASIZE
    float xmax = Px[0];
	float ymax = Py[0];
	float zmax = Pz[0]; 
    float xmin = Px[0];
	float ymin = Py[0];
	float zmin = Pz[0];
	xnorm = (float *)calloc(sizeof(float), length_data);
    ynorm = (float *)calloc(sizeof(float), length_data);
    znorm = (float *)calloc(sizeof(float), length_data);
    for(int i=1; i<length_data; i++){
		if(Px[i] > xmax)      xmax = Px[i];
		else if(Px[i] < xmin) xmin = Px[i];
		if(Py[i] > ymax)      ymax = Py[i];
		else if(Py[i] < ymin) ymin = Py[i];
		if(Pz[i] > zmax)      zmax = Pz[i];
		else if(Pz[i] < zmin) zmin = Pz[i];
	}
	
	for(int i=0; i<length_data; i++){
		xnorm[i] = (Px[i]/abs(xmax-xmin)) * X_RANGE;
		ynorm[i] = (Py[i]/abs(ymax-ymin)) * X_RANGE;
		znorm[i] = (Pz[i]/abs(zmax-zmin)) * X_RANGE;
	}
    
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(display);
    glutReshapeFunc(reshapeCallback);
    glutKeyboardFunc(keyCallback);
    glutMouseFunc(mouseCallback);
    glutMotionFunc(mouseMotion);
	glutMainLoop();
	return 0;
}
