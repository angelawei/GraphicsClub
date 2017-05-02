// Provides a simple 3D vector class. Vector operations can be done using member
// functions, which return new vectors, or static functions, which reuse
// existing vectors to avoid generating garbage.
function Vector(x, y, z) {
  this.x = x || 0;
  this.y = y || 0;
  this.z = z || 0;
}

// ### Instance Methods
// The methods `add()`, `subtract()`, `multiply()`, and `divide()` can all
// take either a vector or a number as an argument.
Vector.prototype = {
  equals: function(v) {
    return this.x == v.x && this.y == v.y && this.z == v.z;
  },
  dot: function(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  },
  length: function() {
    return Math.sqrt(this.dot(this));
  },
  min: function() {
    return Math.min(Math.min(this.x, this.y), this.z);
  },
  max: function() {
    return Math.max(Math.max(this.x, this.y), this.z);
  },
  toAngles: function() {
    return {
      theta: Math.atan2(this.z, this.x),
      phi: Math.asin(this.y / this.length())
    };
  },
  angleTo: function(a) {
    return Math.acos(this.dot(a) / (this.length() * a.length()));
  },
  toArray: function(n) {
    return [this.x, this.y, this.z].slice(0, n || 3);
  },
  set: function(x, y, z) {
    this.x = x; this.y = y; this.z = z;
    return this;
  }
};

// ### Static Methods
// `Vector.randomDirection()` returns a vector with a length of 1 and a
// statistically uniform direction. `Vector.lerp()` performs linear
// interpolation between two vectors.
Vector.negative = function(a, b) {
  b.x = -a.x; b.y = -a.y; b.z = -a.z;
  return b;
};
Vector.add = function(a, b, c) {
  if (b instanceof Vector) { c.x = a.x + b.x; c.y = a.y + b.y; c.z = a.z + b.z; }
  else { c.x = a.x + b; c.y = a.y + b; c.z = a.z + b; }
  return c;
};
Vector.subtract = function(a, b, c) {
  if (b instanceof Vector) { c.x = a.x - b.x; c.y = a.y - b.y; c.z = a.z - b.z; }
  else { c.x = a.x - b; c.y = a.y - b; c.z = a.z - b; }
  return c;
};
Vector.multiply = function(a, b, c) {
  if (b instanceof Vector) { c.x = a.x * b.x; c.y = a.y * b.y; c.z = a.z * b.z; }
  else { c.x = a.x * b; c.y = a.y * b; c.z = a.z * b; }
  return c;
};
Vector.divide = function(a, b, c) {
  if (b instanceof Vector) { c.x = a.x / b.x; c.y = a.y / b.y; c.z = a.z / b.z; }
  else { c.x = a.x / b; c.y = a.y / b; c.z = a.z / b; }
  return c;
};
Vector.pow = function(a, b, c) {
  c.x = Math.pow(a.x, b); c.y = Math.pow(a.y, b); c.z = Math.pow(a.z, b);
  return c;
};
Vector.cross = function(a, b, c) {
  var ax = a.x, ay = a.y, az = a.z;
  var bx = b.x, by = b.y, bz = b.z;
  c.x = ay * bz - az * by;
  c.y = az * bx - ax * bz;
  c.z = ax * by - ay * bx;
  return c;
};
Vector.normalize = function(a, b) {
  var length = a.length();
  if (length > 0) { b.x = a.x / length; b.y = a.y / length; b.z = a.z / length; }
  else { b.x = a.x; b.y = a.x; b.z = a.z }
  return b;
};
Vector.fromAngles = function(theta, phi, a) {
  a.x = Math.cos(theta) * Math.cos(phi);
  a.y = Math.sin(phi);
  a.z = Math.sin(theta) * Math.cos(phi);
  return a;
};
Vector.randomDirection = function(a) {
  return Vector.fromAngles(Math.random() * Math.PI * 2, Math.asin(Math.random() * 2 - 1), a);
};
Vector.reflect = function(a, normal, b) {
  // d − 2 * d.dot(n) * n
  return Vector.subtract(a, Vector.multiply(normal, 2 * a.dot(normal), b), b);
};
Vector.min = function(a, b, c) {
  if (b instanceof Vector) { c.x = Math.min(a.x, b.x); c.y = Math.min(a.y, b.y); c.z = Math.min(a.z, b.z); }
  else { c.x = Math.min(a.x, b); c.y = Math.min(a.y, b); c.z = Math.min(a.z, b); }
  return c;
};
Vector.max = function(a, b, c) {
  if (b instanceof Vector) { c.x = Math.max(a.x, b.x); c.y = Math.max(a.y, b.y); c.z = Math.max(a.z, b.z); }
  else { c.x = Math.max(a.x, b); c.y = Math.max(a.y, b); c.z = Math.max(a.z, b); }
  return c;
};
Vector.clamp = function(a, minVal, maxVal, b) {
  // min(max(a, minVal), maxVal)
  return Vector.min(Vector.max(a, minVal, b), maxVal, b);
};
Vector.lerp = function(a, b, fraction, c) {
  // (b - a) * fraction + a
  return Vector.add(Vector.multiply(Vector.subtract(b, a, c), fraction, c), a, c);
};
Vector.fromArray = function(a, b) {
  b.x = a[0]; b.y = a[1]; b.z = a[2];
  return b;
};
Vector.copy = function(a, b) {
  b.x = a.x; b.y = a.y; b.z = a.z;
  return b;
};
Vector.clone = function(a) {
  return new Vector(a.x, a.y, a.z);
};

// __________________________________________

function Ray(origin, direction) {
  this.origin = origin ? Vector.clone(origin) : new Vector();
  this.direction = direction ? Vector.clone(direction) : new Vector();
};

Ray.prototype = {
  set: function(origin, direction) {
    Vector.copy(origin, this.origin);
    Vector.copy(direction, this.direction);
    return this;
  },
  copy: function(ray) {
    return this.set(ray.origin, ray.direction);
  }
};

var REFLECTION_TYPES = {
  DIFFUSE: 0,
  SPECULAR: 1,
  REFRACTIVE: 2
};

// shallow copy because this constructor is only ever used in objects
function Sphere(center, radius, color, emission, reflectionType) {
  this.center = center;
  this.radius = radius;
  this.color = color;
  this.emission = emission;
  this.reflectionType = reflectionType; // DIFFUSE, SPECULAR, REFRACTIVE
};

var NUM_OBJECTS = 9;
var objects = [
  // left wall
  new Sphere(
    new Vector(1e5+1.0,40.8,81.6),
    1e5,
    new Vector(0.75,0.25,0.25),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // right wall
  new Sphere(
    new Vector(-1e5+99.0,40.8,81.6),
    1e5,
    new Vector(0.25,0.25,0.75),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // back wall
  new Sphere(
    new Vector(50.0,40.8, 1e5),
    1e5,
    new Vector(0.75,0.75,0.75),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // front wall
  new Sphere(
    new Vector(50.0,40.8,-1e5+170.0),
    1e5,
    new Vector(0.0,0.0,0.0),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // bottom wall
  new Sphere(
    new Vector(50.0, 1e5, 81.6),
    1e5,
    new Vector(0.75,0.75,0.75),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // top wall
  new Sphere(
    new Vector(50.0,-1e5+81.6,81.6),
    1e5,
    new Vector(0.75,0.75,0.75),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.DIFFUSE
  ),
  // mirror ball
  new Sphere(
    new Vector(27.0,16.5,47.0),
    16.5,
    new Vector(1.0,1.0,1.0),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.SPECULAR
  ),
  // glass ball
  new Sphere(
    new Vector(73.0,16.5,78.0),
    16.5,
    new Vector(0.8, 1.0, 0.95),
    new Vector(0.0,0.0,0.0),
    REFLECTION_TYPES.REFRACTIVE
  ),
  // top light
  new Sphere(
    new Vector(50.0,681.33,81.6),
    600.0,
    new Vector(0.0,0.0,0.0),
    new Vector(12.0,12.0,12.0),
    REFLECTION_TYPES.DIFFUSE
  )
];

var tmpVector = new Vector();

var EPSILON = 0.001;
function sphereIntersect(sphere, ray) {
  // Solve for t (distance from ray origin)
  // t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0

  // originToPosition = sphere.center - ray.origin;
  var originToPosition = Vector.subtract(sphere.center, ray.origin, tmpVector);
  var b = originToPosition.dot(ray.direction);
  var determinant = (b * b) - originToPosition.dot(originToPosition) + sphere.radius * sphere.radius;

  if (determinant < 0.0) return -1.0; // ray misses

  determinant = Math.sqrt(determinant);

  var t = 0.0;
  if ((t = b - determinant) > EPSILON)
    return t;
  else if ((t = b + determinant) > EPSILON)
    return t;
  return -1.0;
}

function sceneIntersect(ray, closest, selfId) {
  var dist;
  closest.dist = Number.POSITIVE_INFINITY;
  closest.id = -1;

  for(var i = 0; i < NUM_OBJECTS; i++) {
    dist = sphereIntersect(objects[i], ray);

    if (i != selfId && dist > 0.0 && dist < closest.dist) {
      closest.id = i;
      closest.dist = dist;
    }
  }

  return closest.dist < Number.POSITIVE_INFINITY;
}

var SAMPLES = 1;
var MAX_DEPTH = 5;
var MIN_DEPTH = 3;

function radiance(cameraRay) {
  // TODO make sure these work with recursion
  var ray = new Ray();
  var hitPosition = new Vector();
  var hitNormal = new Vector();
  var newRayDirection = new Vector();
  var accumulatedColor = new Vector();
  var accumulatedReflectance = new Vector();
  var hit = { dist: 0, id: 0 };

  ray.copy(cameraRay);

  hit.dist = Number.POSITIVE_INFINITY;
  hit.id = -1;

  accumulatedColor.set(0.0, 0.0, 0.0);
  accumulatedReflectance.set(1.0, 1.0, 1.0);

  var didHit;
  for (var depth = 0; depth < MAX_DEPTH; ++depth) {
    didHit = sceneIntersect(ray, hit, hit.id);
    if (!didHit) break;

    var hitObject = objects[hit.id];
    var hitReflectance = hitObject.color;

    // accumulatedColor += hitObject.emission * accumulatedReflectance
    Vector.add(accumulatedColor, Vector.multiply(hitObject.emission, accumulatedReflectance, tmpVector), accumulatedColor);

    // russian roulette
    var maxReflectance = hitReflectance.max();
    if (depth > MIN_DEPTH) {
      if (Math.random() < maxReflectance) {
        // hitReflectance /= maxReflectance
        Vector.divide(hitReflectance, maxReflectance, hitReflectance);
      } else {
        break;
      }
    }

    // accumulatedReflectance *= hitReflectance
    Vector.multiply(accumulatedReflectance, hitReflectance, accumulatedReflectance);

    // hitPosition = ray.origin + ray.direction * hitDist
    Vector.add(ray.origin, Vector.multiply(ray.direction, hit.dist, hitPosition), hitPosition);
    // hitNormal = normalize(hitPosition - hitObject.center);
    Vector.normalize(Vector.subtract(hitPosition, hitObject.center, hitNormal), hitNormal);

    var dDotN = -hitNormal.dot(ray.direction);
    var into = dDotN > 0.0 ? 1.0 : -1.0;
    dDotN = Math.abs(dDotN);

    if (hitObject.reflectionType == REFLECTION_TYPES.DIFFUSE) {
      Vector.multiply(hitNormal, into, hitNormal);
      var diffuseDirection = Vector.randomDirection(newRayDirection);
      // flip diffuseDirection over hit surface if diffuseDirection is not pointing out of hit surface
      var ddDotHn = hitNormal.dot(diffuseDirection);
      if (ddDotHn <= 0.0) {
        // diffuseDirection -= 2 * hitNormal.dot(diffuseDirection) * hitNormal
        Vector.subtract(diffuseDirection, Vector.multiply(hitNormal, 2 * ddDotHn, tmpVector), diffuseDirection);
      }

      ray.set(hitPosition, diffuseDirection);
    } else if (hitObject.reflectionType == REFLECTION_TYPES.SPECULAR) {
      ray.set(hitPosition, Vector.reflect(ray.direction, hitNormal, newRayDirection));
    } else {
      var IOR_AIR = 1.0;
      var IOR_GLASS = 1.5;
      var ratioIor = (into > 0.0) ? IOR_AIR / IOR_GLASS : IOR_GLASS / IOR_AIR;
      var cos2t = 1.0 - ratioIor * ratioIor * (1.0 - dDotN * dDotN);

      ray.set(hitPosition, Vector.reflect(ray.direction, hitNormal, newRayDirection));
      if (cos2t < 0.0) continue;  // Total internal reflection

      // refractedDirection = normalize(ray.direction * ratioIor - into * hitNormal * (dDotN * ratioIor + sqrt(cos2t)));
      Vector.multiply(ray.direction, ratioIor, newRayDirection);
      Vector.subtract(newRayDirection, Vector.multiply(hitNormal, into * dDotN * ratioIor + Math.sqrt(cos2t), tmpVector), newRayDirection);
      var refractedDirection = Vector.normalize(newRayDirection, newRayDirection);

      var a = IOR_GLASS - IOR_AIR;
      var b = IOR_GLASS + IOR_AIR;
      var R0 = a * a / (b * b); // reflection coefficient at normal incidence
      var c = 1.0 - ((into > 0.0) ? dDotN : refractedDirection.dot(hitNormal)); // 1 - cos(theta)
      var reflectionCoeff = R0 + (1.0 - R0) * c*c*c*c*c; // Schlick's approximation

      // if branching it could have been done without probability:
      // radianceReflection * reflectionCoeff + radianceRefraction * (1 - reflectionCoeff)

      var probability = 0.25 + 0.5 * reflectionCoeff;
      var reflectionProb = reflectionCoeff / probability;
      var refractionProb = (1.0 - reflectionCoeff) / (1.0 - probability);
      if (Math.random() < probability) {
        // accumulatedReflectance *= reflectionProb;
        Vector.multiply(accumulatedReflectance, reflectionProb, accumulatedReflectance);
      } else {
        // accumulatedReflectance *= refractionProb;
        Vector.multiply(accumulatedReflectance, refractionProb, accumulatedReflectance);
        ray.set(hitPosition, refractedDirection);
      }
    }
  }

  return accumulatedColor;
}

// __________________________________________



var c = document.getElementById('c'),
  width = 320,
  height = 240;

// Get a context in order to generate a proper data array. We aren't going to
// use traditional Canvas drawing functions like `fillRect` - instead this
// raytracer will directly compute pixel data and then put it into an image.
c.width = width;
c.height = height;
var ctx = c.getContext('2d'),
  data = ctx.getImageData(0, 0, width, height);

var camera = new Ray(
  new Vector(50.0, 40.8, 169.0),
  new Vector(0.0, -0.00915293, -0.999958)
);

var prevColor = new Float32Array(width * height * 4);
function render() {
  var index;
  var color = new Vector();
  var pixelRay = new Ray();
  Vector.copy(camera.origin, pixelRay.origin);

  var cx = new Vector(1.0, 0.0, 0.0), cy = new Vector();
  // cy = normalize(cross(cx, camera.direction));
  Vector.normalize(Vector.cross(cx, camera.direction, cy), cy);
  // cx = cross(camera.direction, cy);
  Vector.cross(camera.direction, cy, cx);

  for (var x = 0; x < width; x++) {
    for (var y = 0; y < height; y++) {
      var u = 2.0 * x / width - 1.0;
      var v = 2.0 * (1.0 - y / height) - 1.0;

      // pixelDirection = normalize(camera.direction + 0.53135 * (width/height * u * cx + v * cy));
      var pixelDirection = pixelRay.direction;
      Vector.multiply(cx, width/height * u, pixelDirection);
      Vector.multiply(cy, v, tmpVector);
      Vector.add(pixelDirection, tmpVector, pixelDirection);
      Vector.multiply(pixelDirection, 0.53135, pixelDirection);
      Vector.add(camera.direction, pixelDirection, pixelDirection);
      Vector.normalize(pixelDirection, pixelDirection);

      // color.set(0.0, 0.0, 0.0);
      // for (var i = 0; i < SAMPLES; ++i) {
      //   // color += radiance(pixelRay)
      //   Vector.add(color, radiance(pixelRay), color);
      // }
      // Vector.divide(color, SAMPLES, color);
      // Vector.clamp(color, 0.0, 1.0, color);
      // Vector.pow(color, 1/2.2, color);

      // index = (x * 4) + (y * width * 4);
      // data.data[index + 0] = Math.round(color.x * 255);
      // data.data[index + 1] = Math.round(color.y * 255);
      // data.data[index + 2] = Math.round(color.z * 255);
      // data.data[index + 3] = 255;

      // Moving average / Multipass
      Vector.copy(radiance(pixelRay), color);
      color.x = (prevColor[index + 0] * frameNumber + color.x) / (frameNumber + 1);
      color.y = (prevColor[index + 1] * frameNumber + color.y) / (frameNumber + 1);
      color.z = (prevColor[index + 2] * frameNumber + color.z) / (frameNumber + 1);
      prevColor[index + 0] = color.x;
      prevColor[index + 1] = color.y;
      prevColor[index + 2] = color.z;

      Vector.clamp(color, 0.0, 1.0, color);
      Vector.pow(color, 1/2.2, color);

      index = (x * 4) + (y * width * 4);
      data.data[index + 0] = Math.round(color.x * 255);
      data.data[index + 1] = Math.round(color.y * 255);
      data.data[index + 2] = Math.round(color.z * 255);
      data.data[index + 3] = 255;
    }
  }

  ctx.putImageData(data, 0, 0);
}

var frameNumber = 0;
function tick() {
  render();
  console.log(frameNumber);
  frameNumber++;

  if (playing) setTimeout(tick, 10);
}

var playing = false;

function play() {
  playing = true;
  tick();
}

function stop() {
  playing = false;
}

// Then let the user control a cute playing animation!
document.getElementById('play').onclick = play;
document.getElementById('stop').onclick = stop;