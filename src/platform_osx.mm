#import <Cocoa/Cocoa.h>
#import <OpenGL/OpenGL.h>
#import <CoreVideo/CVDisplayLink.h>

#include <stdint.h>
#include <mach/mach_time.h>
#include <libproc.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "ao.h"

// NOTES:
//  + window resizing is currently disabled because Cocoa's NSOpenGLView
//    breaks the normal glViewport behaviour
//  + initial window size can temporarily be defined with the corresponding
//    defines
#define OSX_ALLOW_WINDOW_RESIZE 0
#define OSX_INITIAL_WINDOW_WIDTH 800
#define OSX_INITIAL_WINDOW_HEIGHT 600

static CVReturn global_display_link_callback(CVDisplayLinkRef display_link, const CVTimeStamp* now, const CVTimeStamp* output_time, CVOptionFlags flags_in, CVOptionFlags* flags_out, void* display_link_context);

@interface FileDialogReturn : NSObject {
@public
	NSURL* url;
}
@end

@implementation FileDialogReturn
@end

@interface View : NSOpenGLView<NSWindowDelegate> {
@public
	CVDisplayLinkRef display_link;
    int32_t is_initialized;
}
- (NSPoint)getWindowOrigin;
- (void)openFileDialog:(FileDialogReturn*)result;
- (void)saveFileDialog:(FileDialogReturn*)result;
@end

static ao_memory_t mem;
static View* view;
static mach_timebase_info_data_t mach_clock_frequency;

loaded_file_t platform_load_file(const char* filename)
{
	loaded_file_t result = {0, 0};
    FILE* file = fopen(filename, "r");
    if (file) {
    	std::cout << filename << std::endl;
        fseek(file, 0, SEEK_END);
        int32_t file_size = ftell(file);
        rewind(file);
		if (file_size > 0) {
			void* data = calloc((size_t)file_size, 1);
			if (data) {
				int32_t read_size = fread(data, 1, file_size, file);
				if (read_size == file_size) {
					result.size = file_size;
					result.contents = data;
				}
				else {
					free(data);
				}
			}
		}
		fclose(file);
    }
    return (result);
}

double get_milliseconds(void)
{
	return (double)(mach_absolute_time() * (mach_clock_frequency.numer / mach_clock_frequency.denom) / 1000000.0);
}

@implementation View
- (id)initWithFrame:(NSRect)frame
{
    is_initialized = 0;

	NSOpenGLPixelFormatAttribute pixel_format_attributes[] = {
		NSOpenGLPFADoubleBuffer,
		NSOpenGLPFAAccelerated,
		NSOpenGLPFAColorSize, 32,
		NSOpenGLPFAAlphaSize, 8,
		NSOpenGLPFADepthSize, 24,
		NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core,
		0
	};
	NSOpenGLPixelFormat* pixel_format = [[NSOpenGLPixelFormat alloc] initWithAttributes:pixel_format_attributes];
	if (!pixel_format) {
		return (nil);
	}
	self = [super initWithFrame:frame pixelFormat:[pixel_format autorelease]];
	return (self);
}

- (void)prepareOpenGL
{
	[super prepareOpenGL];

	[[self window] setLevel:NSNormalWindowLevel];
	[[self window] makeKeyAndOrderFront:self];

	[[self openGLContext] makeCurrentContext];
	GLint swap_interval = 1;
	[[self openGLContext] setValues:&swap_interval forParameter:NSOpenGLCPSwapInterval];

	CVDisplayLinkCreateWithActiveCGDisplays(&display_link);
	CVDisplayLinkSetOutputCallback(display_link, &global_display_link_callback, self);

	CGLContextObj cgl_context = (CGLContextObj)[[self openGLContext] CGLContextObj];
	CGLPixelFormatObj pixel_format = (CGLPixelFormatObj)[[self pixelFormat] CGLPixelFormatObj];
	CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext(display_link, cgl_context, pixel_format);

	GLint window_size[2] = {mem.window_width, mem.window_height};
	CGLSetParameter(cgl_context, kCGLCPSurfaceBackingSize, window_size);
	CGLEnable(cgl_context, kCGLCESurfaceBackingSize);

	CVDisplayLinkStart(display_link);
	return;
}

- (BOOL)acceptsFirstResponder
{
	return (YES);
}

static void OnMouseMoveEvent(NSEvent* event)
{
	NSPoint point = [view convertPoint:[event locationInWindow] fromView:nil];
	return;
}

- (void)mouseMoved:(NSEvent*)event
{
	OnMouseMoveEvent(event);
	return;
}

- (void)mouseDragged:(NSEvent*)event
{
	OnMouseMoveEvent(event);
	return;
}

- (void)mouseDown:(NSEvent*)event
{
	//
}

- (void)mouseUp:(NSEvent*)event
{
	//
}

- (void)rightMouseDown:(NSEvent*)event
{
	//
}

- (void)rightMouseUp:(NSEvent*)event
{
	//
}

- (void)otherMouseDown:(NSEvent*)event
{
	//
}

- (void)otherMouseUp:(NSEvent*)event
{
	//
}

- (void)mouseEntered:(NSEvent*)event
{
	//
}

- (void)mouseExited:(NSEvent*)event
{
	//
}

- (void)keyDown:(NSEvent*)event
{
	int32_t is_q_key_down = (tolower([[event charactersIgnoringModifiers] characterAtIndex:0]) == 'q');
	if (is_q_key_down) {
		exit(0);
	}
	return;
}

- (void)keyUp:(NSEvent*)event
{
	//
}

- (void)flagsChanged:(NSEvent*)event
{
	uint32 modifiers = [event modifierFlags];

	int32_t is_ctrl_down = ((modifiers & NSControlKeyMask) || (modifiers & NSCommandKeyMask));
	int32_t is_shift_down = ((modifiers & NSShiftKeyMask) != 0);
	int32_t is_alt_down = ((modifiers & NSAlternateKeyMask) != 0);

	return;
}

- (CVReturn)getFrameForTime:(const CVTimeStamp*)OutputTime
{
	[[self openGLContext] makeCurrentContext];
	CGLLockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);

    if (!is_initialized) {
        ao_init(&mem);
        mem.is_running = 1;
        is_initialized = 1;
    }

	// TODO: update frame here
    ao_update_frame(&mem);

    if (!mem.is_running) {
        exit(1);
    }

	float current_time = (float)(mach_absolute_time() * (mach_clock_frequency.numer / mach_clock_frequency.denom) / 1000000000.0f);

	// TODO: sleep here to enforce frame rate

	CGLFlushDrawable((CGLContextObj)[[self openGLContext] CGLContextObj]);

	CGLUnlockContext((CGLContextObj)[[self openGLContext] CGLContextObj]);

	return kCVReturnSuccess;
}

#if OSX_ALLOW_WINDOW_RESIZE
- (void)windowDidResize:(NSNotification*)notification
{
	NSSize frame_size = [[_window contentView] frame].size;
	mem.window_width = frame_size.width;
	mem.window_height = frame_size.height;
	return;
}
#endif

- (void)resumeDisplayRenderer
{
	CVDisplayLinkStop(display_link);
	return;
}

- (void)haltDisplayRenderer
{
	CVDisplayLinkStop(display_link);
	return;
}

- (void)windowWillClose:(NSNotification*)notification
{
	CVDisplayLinkStop(display_link);
	CVDisplayLinkRelease(display_link);
	[NSApp terminate:self];
	return;
}

- (BOOL)isFlipped
{
	// NOTE: set the upper-left corner as the origin for mouse coordinates
	return (YES);
}

- (NSPoint)getWindowOrigin
{
	NSPoint origin = [_window convertRectToScreen:[[_window contentView] frame]].origin;
	return (origin);
}

- (void)openFileDialog:(FileDialogReturn*)result
{
	result->url = nil;
	NSOpenPanel* panel = [NSOpenPanel openPanel];
	[panel makeKeyAndOrderFront:self];
	int32_t panel_run_result = [panel runModal];
	if (panel_run_result == NSFileHandlingPanelOKButton) {
		NSArray* urls = [panel URLs];
		if ([urls count] > 0) {
			result->url = [urls objectAtIndex:0];
		}
	}
	return;
}

- (void)saveFileDialog:(FileDialogReturn*)result
{
	result->url = nil;
	NSSavePanel* panel = [NSSavePanel savePanel];
	[panel makeKeyAndOrderFront:self];
	int32_t panel_run_result = [panel runModal];
	if (panel_run_result == NSFileHandlingPanelOKButton) {
		result->url = [[panel URL] copy];
	}
	return;
}

- (void)dealloc
{
	[super dealloc];
	return;
}
@end

static CVReturn global_display_link_callback(CVDisplayLinkRef display_link, const CVTimeStamp* now, const CVTimeStamp* output_time, CVOptionFlags flags_in, CVOptionFlags* flags_out, void* display_link_context)
{
	CVReturn result = [(View*)display_link_context getFrameForTime:output_time];
	return result;
}

int main(int argc, char* argv[])
{
	mach_clock_frequency.numer = 0;
	mach_clock_frequency.denom = 0;
	mach_timebase_info(&mach_clock_frequency);

	char path_buffer[PATH_MAX];
	proc_pidpath(getpid(), path_buffer, sizeof(path_buffer) - 1);
	if (path_buffer[0])
	{
		char *last_slash = strrchr(path_buffer, '/');
		if (last_slash != NULL) { *last_slash = '\0'; }
		chdir(path_buffer);
	}

	NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];

	[NSApplication sharedApplication];

	NSRect screen_dimensions = [[NSScreen mainScreen] frame];
	int32_t screen_width = screen_dimensions.size.width;
	int32_t screen_height = screen_dimensions.size.height;

	mem.window_width = OSX_INITIAL_WINDOW_WIDTH;
	mem.window_height = OSX_INITIAL_WINDOW_HEIGHT;
	NSRect window_rect = NSMakeRect((screen_width - mem.window_width) / 2, (screen_height - mem.window_height) / 2, mem.window_width, mem.window_height);

#if OSX_ALLOW_WINDOW_RESIZE
	NSUInteger window_style = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask;
#else
	NSUInteger window_style = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask;
#endif
	NSWindow* window = [[NSWindow alloc] initWithContentRect:window_rect styleMask:window_style backing:NSBackingStoreBuffered defer:NO];
	[window autorelease];

	// NOTE: activation policy needs to be set unless XCode is used to build the project
	[NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

	id menu_bar = [[NSMenu new] autorelease];
	id app_menu_item = [[NSMenuItem new] autorelease];
	[menu_bar addItem:app_menu_item];
	[NSApp setMainMenu:menu_bar];

	id app_menu = [[NSMenu new] autorelease];
	id app_name = [[NSProcessInfo processInfo] processName];
	id quit_title = [@"Quit " stringByAppendingString:app_name];
	id quit_menu_item = [[[NSMenuItem alloc] initWithTitle:quit_title action:@selector(terminate:) keyEquivalent:@"q"] autorelease];
	[app_menu addItem:quit_menu_item];
	[app_menu_item setSubmenu:app_menu];

	view = [[[View alloc] initWithFrame:window_rect] autorelease];

	[window setAcceptsMouseMovedEvents:YES];
	[window setContentView:view];
	[window setDelegate:view];
	[window setTitle:app_name];

#if OSX_ALLOW_WINDOW_RESIZE
	// enables osx fullscreen ability
	// NOTE: only works if NSResizableWindowMask is set
	[window setCollectionBehavior: NSWindowCollectionBehaviorFullScreenPrimary];
#endif

	// bring window to front of other windows
	[window orderFrontRegardless];

	// bring window into focus
	[NSApp activateIgnoringOtherApps:true];

	[NSApp run];

	[pool drain];
	return (0);
}

