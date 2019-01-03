// Author:Lee Eul
// Title:PBR RayMarching
//test this code on http://editor.thebookofshaders.com/

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

#define PI 3.141592
#define EPS 0.01
#define MAX 100.



/* **********Control Box********** */
#define DIR_LI 0
#define SHADOW 0

const int lNum = 2;
vec3 larray[lNum];
void lPosSet(){
    larray[0] = vec3(10.);
    larray[1] = vec3(-10.);
}
float lightPower = 1000.;
vec3 lightColor = vec3(1.);
float amb = .05; // 기본 ambient조명

float metallic = .8; // 0~1 직접조정을 원하면 306줄 코드를 주석처리 하세요
float roughness = 0.6; // 0~1 직접조정을 원하면 307줄 코드를 주석처리 하세요
float minSpec = 0.04; // 0~0.09 minSpec은 프레넬이 가장 약할때 해당 매질이 갖는 스페큘러 세기를 결정해주는 역할이다. 만약 금속성이 하나도 없는 매질의 스페큘러 세기를 강하게 해주고 싶으면 수치를 올리세요.
vec3 albedo = vec3(1.,.7, 0.2);
vec3 bgColor = vec3(.2);
/* ******************** */



float u(float f1, float f2){ return min(f1, f2); }
float n(float f1, float f2){ return max(f1, f2); }
float d(float f1, float f2){ return max(f1, -f2); }

float boxSDF( vec3 ray, vec3 loc, vec3 size ){
    ray -= loc;
    vec3 d = abs(ray) - size;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float sphereSDF(vec3 ray, vec3 loc, float f){
    return distance(ray, loc)-f;
}

float SDF(vec3 ray){
    ray /= 1.5; // scaling
    
    float s1 = sphereSDF(ray, vec3(0.), 1.);
    float b1 = boxSDF(ray, vec3(0.), vec3(.8));
    float s2 = sphereSDF(ray, vec3(0.), 0.924);
    
    float b2 = boxSDF(ray, vec3(0.), vec3(0.512));
    float s3 = sphereSDF(ray, vec3(0.), 0.6);
    
    float result = n(s1, b1);
    result = d(result, s2);
    
    result = u(result, n(b2, s3));
    
    return result;
}

float intersect(vec3 cam, vec3 dir){
    vec3 ray;
    float md = 0.;
    
    for(int i=0; i<100; i++){
        ray = cam + dir*md;
        float d = SDF(ray);
        if(d<EPS){
            return md;
        }
        
        md += d;
        if(md>MAX){
            return MAX;
        }
    }
    return MAX;
}

vec3 getN(vec3 ray){
    vec3 v;
    v.x = SDF(ray)-SDF(ray-vec3(EPS, 0., 0.));
    v.y = SDF(ray)-SDF(ray-vec3(0., EPS, 0.));
    v.z = SDF(ray)-SDF(ray-vec3(0., 0., EPS));
    return normalize(v);
}

vec3 fresnel_factor(in vec3 f0, in float product){
    return mix(f0, vec3(1.0), pow(1.01 - product, 5.0));
}

float D_GGX(in float roughness, in float NdH){
    float m = roughness * roughness;
    float m2 = m * m;
    float d = (NdH * m2 - NdH) * NdH + 1.0;
    return m2 / (PI * d * d);
}

float G_schlick(in float roughness, in float NdV, in float NdL){
    float k = roughness * roughness * 0.5;
    float V = NdV * (1.0 - k) + k;
    float L = NdL * (1.0 - k) + k;
    return 0.25 / (V * L);
}

vec3 cooktorrance_specular(in float NdL, in float NdV, in float NdH, in vec3 specular, in float roughness){
    float D = D_GGX(roughness, NdH);
    float G = G_schlick(roughness, NdV, NdL);
    
    float rim = mix(1.0 - roughness * .5 * 0.9, 1.0, NdV);
    return (1.0 / rim) * specular * G * D;
}

float phong_diffuse(){
    return (1.0 / PI);
}

float shadow(vec3 p, vec3 light, float k){
    #if SHADOW
#if DIR_LI
    vec3 dir = normalize(light);
    float max = 10.;
#else
    vec3 dir = normalize(light-p);
    float max = length(light - p);
#endif
    
    vec3 ray;
    float minDist = EPS; // 디테일한 쉐입이 많은 도형의 경우 이 오프셋을 키워줄 필요가 있음
    float penumbra = 1.;
    for(int i=0; i<100; i++){
        ray = p + dir*minDist;
        
        float d = SDF(ray);
        if(d<0.0000008){ //이 수치를 상당히 엄격하게 작게 해야…
            return 0.;
        }
        
        penumbra = min(penumbra, k*d/minDist);
        minDist += d;
        
        if(minDist>=max){
            break;
        }
    }
    return penumbra;
    #else
    return 1.;
    #endif
}


vec3 lightCal(vec3 base, vec3 cam, vec3 ray, vec3 li){
#if DIR_LI
    vec3 L = normalize(li);
    float sha = shadow(ray, L, 10.);
    float A = lightPower*sha;
#else //point light
    vec3 lDist = li-ray;
    vec3 L = normalize(lDist);
    float sha = shadow(ray, li, 10.);
    float A = lightPower/dot(lDist, lDist)*sha;
#endif

    vec3 V = normalize(cam - ray);
    vec3 H = normalize(L+V);
    vec3 nn = getN(ray);
    
    float NdL = max(0.0, dot(nn, L));
    float NdV = max(0.001, dot(nn, V));
    float NdH = max(0.001, dot(nn, H));
    float HdV = max(0.001, dot(H, V));
    
    vec3 specular = mix(vec3(minSpec), base, metallic);
    //프레넬이 가장 약할때 (빛벡터와 뷰벡터가 이루는 각이 0도일때) 해당 매질이 갖는 스페큘러 세기를 결정해주는 역할이다. 만약 금속성이 하나도 없으면, 해당 매질이 프레넬이 가장 약할때 갖는 스페큘러 세기는 minSpec이 된다.
    //금속성에 따른 스페큘러의 칼라를 설정해준다. 정반사에서 비금속은 빛 자체를 튕겨내기에 빛의 색 그대로 정반사가 나타나지만, 금속은 종류마다 고유의 정반사 색이 있다. 그러나 프로그래밍에서는 금속의 종류마다의 정반사 색을 일일이 고려하기 귀찮아서 해당 알베도 값으로 금속성 색을 지정합니다.
    vec3 specfresnel = fresnel_factor(specular, HdV);//프레넬 측정
    vec3 specref = cooktorrance_specular(NdL, NdV, NdH, specfresnel, roughness);
    specref *= vec3(NdL);//해당면에 쪼이는 조도측정
    
    vec3 diffref = (vec3(1.0) - specfresnel) * phong_diffuse() * NdL;//에너지보존을 고려한 역프레넬. 프레넬은 디퓨즈에도 영향을 준다. phong_diffuse는 사방팔방으로 난반사 되는 빛 중에서 정확히 카메라 방향으로 들어오는 빛만을 캐치하기 위한 에너지보존식. 또한 해당면에 쪼이는 조도를 측정한다. 무엇보다 디퓨즈는 러프니스와는 관련이 없다. 속에서 산란되는 색깔이고, 대리석을 아무리 갈아도 거울 되지 않고 흰빛은 계속 난다. 디퓨즈는 어떤 모델을 쓰든 다 공통적으로 구하는 공식이 비슷하다. 재질표현에 중요한건 사실 스페큘러란 말이다.
    
    vec3 reflected_light = vec3(0); //스페큘러 변수생성
    vec3 diffuse_light = vec3(amb); //디퓨즈 변수생성. 

    vec3 light_color = lightColor * A; // 빛의 색깔과, 거리에 따른 강도 설정.
    reflected_light += specref * light_color;
    diffuse_light += diffref * light_color;
    
    vec3 result =
    diffuse_light * mix(base, vec3(0.0), metallic) +
    reflected_light; //금속성을 띔으로해서 표면 아래로 흡수되지 못하는 빛을 고려하여 디퓨즈의 세기를 감소시키는것이다.
    
    return result;
}

vec3 lights(vec3 albedo, vec3 cam, vec3 ray){
    vec3 col;
    
    for(int i=0; i<lNum; i++){
        vec3 li = larray[i];
        col += lightCal(albedo,cam,ray,li);
    }
    
    return col;
}

vec3 random(vec3 coord){
    float f1 = fract(sin(dot(coord, vec3(75.7, 65.9, -127.54)))*1e5); // 0~1
    float f2 = fract(sin(dot(coord, vec3(775.7, -965.9, 827.54)))*1e5); // 0~1
    float f3 = fract(sin(dot(coord, vec3(-175.7, 5.9, 227.54)))*1e5); // 0~1
    
    return vec3(f1,f2,f3);
}

float noise( in vec3 x ){
    // grid
    vec3 p = floor(x);
    vec3 w = fract(x);
    
    // quintic interpolant
    vec3 u = w*w*w*(w*(w*6.0-15.0)+10.0);
    
    
    // gradients
    vec3 ga = random( p+vec3(0.0,0.0,0.0) );
    vec3 gb = random( p+vec3(1.0,0.0,0.0) );
    vec3 gc = random( p+vec3(0.0,1.0,0.0) );
    vec3 gd = random( p+vec3(1.0,1.0,0.0) );
    vec3 ge = random( p+vec3(0.0,0.0,1.0) );
    vec3 gf = random( p+vec3(1.0,0.0,1.0) );
    vec3 gg = random( p+vec3(0.0,1.0,1.0) );
    vec3 gh = random( p+vec3(1.0,1.0,1.0) );
    
    // projections
    float va = dot( ga, w-vec3(0.0,0.0,0.0) );
    float vb = dot( gb, w-vec3(1.0,0.0,0.0) );
    float vc = dot( gc, w-vec3(0.0,1.0,0.0) );
    float vd = dot( gd, w-vec3(1.0,1.0,0.0) );
    float ve = dot( ge, w-vec3(0.0,0.0,1.0) );
    float vf = dot( gf, w-vec3(1.0,0.0,1.0) );
    float vg = dot( gg, w-vec3(0.0,1.0,1.0) );
    float vh = dot( gh, w-vec3(1.0,1.0,1.0) );
    
    // interpolation
    return va +
    u.x*(vb-va) +
    u.y*(vc-va) +
    u.z*(ve-va) +
    u.x*u.y*(va-vb-vc+vd) +
    u.y*u.z*(va-vc-ve+vg) +
    u.z*u.x*(va-vb-ve+vf) +
    u.x*u.y*u.z*(-va+vb+vc-vd+ve-vf-vg+vh);
}

mat3 easyCam( vec2 angle ) {
    vec2 t = angle;
    angle.x = t.y*-1.;
    angle.y = t.x;
    vec2 c = cos( angle );
    vec2 s = sin( angle );
    
    return mat3(
        c.y      ,  0.0, -s.y,
        s.y * s.x,  c.x,  c.y * s.x,
        s.y * c.x, -s.x,  c.y * c.x
    );
}

void main(void){
    vec2 coord = gl_FragCoord.xy/u_resolution;
    coord = coord*2.-1.;
    coord.x *= u_resolution.x/u_resolution.y;
    
    vec3 cam = vec3(0., 0., 3.);
    vec3 dir = normalize(vec3(coord, -1.));
    lPosSet();
    
    vec2 m = u_mouse/u_resolution;
    m = m*2. - 1.;
    m*=2.;
    cam *= easyCam(m); 
    dir *= easyCam(m);
    
    float md = intersect(cam, dir);
    if(md>MAX-EPS){
        gl_FragColor = vec4(vec3(bgColor), 1.);
        return;
    }
    
    vec3 ray = cam + dir*md;
    
    metallic = clamp(noise(ray*8.)*5.5 + .5, 0.3, .7);
    roughness = clamp(noise(ray*8.)*5.5 + .5, 0.3, .7);
    
    vec3 col = lights(albedo,cam,ray);
    
    gl_FragColor = vec4(col, 1.);
}
