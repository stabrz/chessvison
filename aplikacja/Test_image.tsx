import React from 'react';
import { View, Image, StyleSheet } from 'react-native';

const TestImage = ({ onImageLoad }: { onImageLoad?: (uri: string) => void }) => {
  const imageUri = require('./test_image.jpg'); // Adjust according to your project's structure

  // Callback to return image URI when the image is loaded
  React.useEffect(() => {
    if (onImageLoad) {
      onImageLoad(imageUri);
    }
  }, [imageUri]);

  return (
    <View style={styles.container}>
      <Image source={imageUri} style={styles.image} resizeMode="contain" />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  image: {
    width: 300, // Adjust width as necessary
    height: 300, // Adjust height as necessary
  },
});

export default TestImage;